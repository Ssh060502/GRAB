"""Finite multi-start trajectory search, not a proof of reachability/infeasibility.

Only the first frame uses unrelated seeds. Later frames use local bounded IK
and optional target subdivision. All winding representatives are kept initially.
No collision or acceleration constraints are imposed.
"""
from itertools import product
from copy import copy
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def turn_variants(q, bounds):
    """Enumerate all in-limit 2pi representatives without merging windings."""
    options = []
    for value, (low, high) in zip(q, bounds):
        first = int(np.ceil((low-value-1e-9)/(2*np.pi)))
        last = int(np.floor((high-value+1e-9)/(2*np.pi)))
        options.append([np.clip(value+2*np.pi*k, low, high) for k in range(first,last+1)])
    return [np.array(values) for values in product(*options)]


def search_arm_trajectory(model, positions, rotations, seeds, orientation_scale=.05,
                          max_step=.3, limit_margin=.001, subdivisions=3,
                          fps=120., progress=None):
    """Return best complete path, or raise without returning a partial trajectory.

    Hard per-frame tolerances: 1 mm and 0.5 degrees. max_step is per-joint radians
    per stored frame, not a velocity limit. Rank complete tracks by required speed
    ratio, then joint steps/path length and limit clearance. Slowdown remains the
    caller's responsibility. Subdivision only assists IK; it adds no output frames.
    """
    if (len(positions)==0 or len(positions)!=len(rotations) or not np.isfinite(
            [orientation_scale,max_step,limit_margin,fps]).all() or orientation_scale<=0 or
            max_step<=0 or limit_margin<0 or fps<=0 or subdivisions<0 or int(subdivisions)!=subdivisions):
        raise ValueError("Invalid trajectory search settings/empty targets")
    bounds = model.limits(model.arm_names)
    low, high = bounds[:,0]+limit_margin, bounds[:,1]-limit_margin
    if np.any(low>=high):
        raise ValueError("Limit margin exceeds joint range")
    position_tol, rotation_tol = .001, np.deg2rad(.5)

    def errors(q, pos, rot):
        achieved=model.palm_pose(q)
        return (np.linalg.norm(achieved[:3,3]-pos),
                Rotation.from_matrix(rot.T@achieved[:3,:3]).magnitude())

    def good(result, pos, rot):
        p,r=errors(result.x,pos,rot)
        return (np.isfinite(result.x).all() and p<=position_tol and r<=rotation_tol
                and np.all(result.x>=low-1e-9) and np.all(result.x<=high+1e-9))

    def solve(pos, rot, initial, local=False):
        lo,hi=low.copy(),high.copy()
        if local:
            lo=np.maximum(lo,initial-max_step)
            hi=np.minimum(hi,initial+max_step)
        def residual(q):
            achieved=model.palm_pose(q)
            return np.r_[achieved[:3,3]-pos, orientation_scale*
                         Rotation.from_matrix(rot.T@achieved[:3,:3]).as_rotvec()]
        return least_squares(residual,np.clip(initial,lo+1e-10,hi-1e-10),bounds=(lo,hi),
                             max_nfev=250,ftol=1e-10,xtol=1e-10,gtol=1e-10)

    starts=[]
    for seed in seeds:
        # Preserve alternate windings instead of canonicalizing the initial angle.
        result=solve(positions[0],rotations[0],np.clip(seed,low,high))
        for q in turn_variants(result.x,bounds):
            candidate=copy(result)
            candidate.x=q
            if good(candidate,positions[0],rotations[0]) and not any(
                    np.max(np.abs(q-old.x))<1e-4 for old in starts):
                starts.append(candidate)
    if not starts:
        raise RuntimeError("Trajectory search found no acceptable first-frame candidate. "
                           "This does not prove infeasibility; inspect placement, seeds and tolerances.")
    if progress:
        progress(f"{model.side}: tracing {len(starts)} first-frame candidates (including distinct turns)")

    def advance(previous,pos,rot,depth):
        result=solve(pos,rot,previous,local=True)
        if good(result,pos,rot):
            return result
        if depth==0:
            return None
        old=model.palm_pose(previous)
        middle_pos=(old[:3,3]+pos)/2
        delta=Rotation.from_matrix(old[:3,:3].T@rot).as_rotvec()
        middle_rot=old[:3,:3]@Rotation.from_rotvec(delta/2).as_matrix()
        middle=advance(previous,middle_pos,middle_rot,depth-1)
        if middle is None:
            return None
        endpoint=advance(middle.x,pos,rot,depth-1)
        if endpoint is None or np.max(np.abs(endpoint.x-previous))>max_step+1e-8:
            return None
        return endpoint

    summaries=[]; best=None
    velocities=model.velocity_limits(model.arm_names)
    if not np.isfinite(velocities).all() or np.any(velocities<=0):
        raise ValueError("Finite positive URDF velocity limits required")
    for index,start in enumerate(starts):
        path=[start.x.copy()]; flags=[bool(start.success)]; failed=-1
        for t in range(1,len(positions)):
            result=advance(path[-1],positions[t],rotations[t],subdivisions)
            if result is None:
                failed=t
                break
            path.append(result.x.copy());flags.append(bool(result.success))
            if progress and t%500==0:
                progress(f"{model.side} candidate {index+1}/{len(starts)}: arm frame {t}/{len(positions)}")
        path=np.asarray(path)
        delta=np.abs(np.diff(path,axis=0))
        clearance=float(np.minimum(path-low,high-path).min()+limit_margin)
        speed_ratio=float((delta*fps/velocities).max()) if len(delta) else 0.
        step=float(delta.max()) if len(delta) else 0.
        summaries.append([failed,len(path),speed_ratio,step,clearance])
        if progress:
            progress(f"{model.side} candidate {index+1}: "+
                     (f"complete; speed ratio {speed_ratio:.2f}, max step {step:.4f} rad" if failed<0
                      else f"stopped at frame {failed}; local tracking/step/clearance criterion failed"))
        if failed<0:
            score=(speed_ratio,step,float(delta.sum()),-clearance)
            if best is None or score<best[0]:
                best=(score,path,np.asarray(flags,dtype=np.uint8),index)
    if best is None:
        failures=[int(row[0]) for row in summaries]
        raise RuntimeError(f"No complete trajectory among {len(starts)} candidates; first failed frames {failures}. "
                           "No output written. This is a finite search failure, NOT a proof of unreachable motion. "
                           "Inspect singularities, tolerances, placement and additional seeds.")
    return dict(path=best[1],success=best[2],selected=best[3],
                starts=np.array([s.x for s in starts]),summary=np.asarray(summaries),
                summary_columns="failed_frame(-1=complete),tracked_frames,speed_ratio,max_step_rad,min_limit_margin_rad")
