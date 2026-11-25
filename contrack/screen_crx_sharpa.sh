#!/usr/bin/env bash
# Screen every GRAB sequence of one object for the CRX-5iA + Sharpa route.
#
# Per sequence: GRAB -> source H5 (grab env), contact filter, MANO keypoints (grab env),
# arm-search probe at the source speed (retarget env), and optionally the final file at
# the smallest whole --time-scale that keeps the arm(s) within the URDF velocity limits.
# One summary line per sequence goes to <out>/summary.txt; details go to <out>/<name>.log.
set -u

usage() {
    cat <<'USAGE'
Usage: contrack/screen_crx_sharpa.sh --object NAME [options]

  --object NAME      GRAB object name, e.g. waterbottle, mug, hammer, fryingpan
                     (matches grab/s*/NAME_*.npz)
  --hands H          right (default), left or both
  --subjects "S..."  only these subjects, e.g. "s1 s5" (default: all)
  --min-contact N    frames each retargeted hand must touch the object (default 30)
  --max-other N      right/left only: frames the other hand may touch it (default 0)
  --finalize         after an OK probe, also write the final H5 at ceil(needed time scale)
  --tmux             run detached in tmux session crx_<object>_<hands>
  --out DIR          default: $GRAB_CONTRACK/crx_sharpa_candidates/<object>_<hands>

Environment overrides (defaults in brackets):
  GRAB_ROOT [/mnt/ssd1/shenghe/datasets/GRAB_dataset]
  GRAB_MODELS [/mnt/ssd1/shenghe/datasets/GRAB_models]
  GRAB_CONTRACK [/mnt/ssd1/shenghe/datasets/GRAB_contrack]
  CRX_RIGHT_URDF / CRX_LEFT_URDF [~/ConTrack/assets/urdf/crx_sharpa_{right,left}.urdf]
  GRAB_ENV [grab]  RETARGET_ENV [retarget]
USAGE
}

ORIGINAL_ARGS=("$@")
OBJECT="" HANDS=right SUBJECTS="" MIN_CONTACT=30 MAX_OTHER=0 FINALIZE=0 USE_TMUX=0 OUT=""
while [ $# -gt 0 ]; do
    case "$1" in
        --object) OBJECT="$2"; shift 2 ;;
        --hands) HANDS="$2"; shift 2 ;;
        --subjects) SUBJECTS="$2"; shift 2 ;;
        --min-contact) MIN_CONTACT="$2"; shift 2 ;;
        --max-other) MAX_OTHER="$2"; shift 2 ;;
        --finalize) FINALIZE=1; shift ;;
        --tmux) USE_TMUX=1; shift ;;
        --out) OUT="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done
[ -n "$OBJECT" ] || { usage >&2; exit 2; }
case "$HANDS" in right|left|both) ;; *) echo "--hands must be right, left or both" >&2; exit 2 ;; esac
[[ "$MIN_CONTACT" =~ ^[0-9]+$ && "$MAX_OTHER" =~ ^[0-9]+$ ]] || { echo "Contact counts must be integers" >&2; exit 2; }

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_DIR=$(dirname "$SCRIPT_DIR")
G=${GRAB_ROOT:-/mnt/ssd1/shenghe/datasets/GRAB_dataset}
M=${GRAB_MODELS:-/mnt/ssd1/shenghe/datasets/GRAB_models}
D=${GRAB_CONTRACK:-/mnt/ssd1/shenghe/datasets/GRAB_contrack}
URDF_R=${CRX_RIGHT_URDF:-$HOME/ConTrack/assets/urdf/crx_sharpa_right.urdf}
URDF_L=${CRX_LEFT_URDF:-$HOME/ConTrack/assets/urdf/crx_sharpa_left.urdf}
GRAB_ENV=${GRAB_ENV:-grab}
RETARGET_ENV=${RETARGET_ENV:-retarget}
CONDA=${CONDA_EXE:-conda}
OUT=${OUT:-$D/crx_sharpa_candidates/${OBJECT}_${HANDS}}

if [ "$USE_TMUX" = 1 ]; then
    session="crx_${OBJECT}_${HANDS}"
    if tmux has-session -t "$session" 2>/dev/null; then
        echo "tmux session $session already exists: tmux attach -t $session" >&2; exit 1
    fi
    args=()
    for a in "${ORIGINAL_ARGS[@]}"; do [ "$a" = "--tmux" ] || args+=("$a"); done
    tmux new-session -d -s "$session" "$(printf '%q ' "$SCRIPT_DIR/screen_crx_sharpa.sh" "${args[@]}"); exec bash"
    echo "Started tmux session $session"
    echo "  watch:   tmux attach -t $session   (leave again: Ctrl+b then d)"
    echo "  summary: $OUT/summary.txt"
    exit 0
fi

for f in "$URDF_R" $([ "$HANDS" = right ] || echo "$URDF_L"); do
    [ -f "$f" ] || { echo "Missing URDF: $f" >&2; exit 1; }
done
shopt -s nullglob
sequences=("$G"/grab/s*/"${OBJECT}"_*.npz)
[ ${#sequences[@]} -gt 0 ] || { echo "No GRAB sequences match $G/grab/s*/${OBJECT}_*.npz" >&2; exit 1; }
mkdir -p "$OUT"
cd "$CODE_DIR"
urdf_args=(--right-urdf "$URDF_R")
[ "$HANDS" = right ] || urdf_args+=(--left-urdf "$URDF_L")

{
    echo "# $(date '+%F %T')  object=$OBJECT hands=$HANDS min_contact=$MIN_CONTACT max_other=$MAX_OTHER finalize=$FINALIZE"
    echo "# contact L/R = frames each hand touches the object inside the cropped clip"
} | tee -a "$OUT/summary.txt"

for npz in "${sequences[@]}"; do
    subject=$(basename "$(dirname "$npz")")
    clip=$(basename "$npz" .npz)
    if [ -n "$SUBJECTS" ] && [[ " $SUBJECTS " != *" $subject "* ]]; then continue; fi
    seq="$subject/$clip"; name="grab-${subject}_${clip}"
    src="$OUT/$name-source.h5"; log="$OUT/$name.log"
    echo "[$(date +%H:%M)] start $seq" >&2

    if ! "$CONDA" run -n "$GRAB_ENV" python contrack/grab_to_contrack_sharpa.py \
            --grab-root "$G" --seq "$seq" --out "$src" > "$log" 2>&1; then
        echo "$seq  FAILED  GRAB conversion (see $log)"; continue
    fi
    frames=$(grep -oE 'cropping \[[0-9]+:[0-9]+\] -> [0-9]+ frames' "$log")
    read -r L R < <("$CONDA" run -n "$RETARGET_ENV" python -c \
        "import h5py,sys; print(*h5py.File(sys.argv[1],'r')['contacts/0/is_contact'][:].any(2).sum(0))" "$src")
    case "$HANDS" in
        right) keep=$([ "${R:-0}" -ge "$MIN_CONTACT" ] && [ "${L:-0}" -le "$MAX_OTHER" ] && echo 1) ;;
        left)  keep=$([ "${L:-0}" -ge "$MIN_CONTACT" ] && [ "${R:-0}" -le "$MAX_OTHER" ] && echo 1) ;;
        both)  keep=$([ "${L:-0}" -ge "$MIN_CONTACT" ] && [ "${R:-0}" -ge "$MIN_CONTACT" ] && echo 1) ;;
    esac
    if [ "${keep:-}" != 1 ]; then
        echo "$seq  $frames  contact L/R=$L/$R  SKIPPED  contact does not fit --hands $HANDS"; continue
    fi

    if ! "$CONDA" run -n "$GRAB_ENV" python contrack/sharpa_keypoints.py --h5 "$src" \
            --grab-root "$G" --model-path "$M" --side "$HANDS" >> "$log" 2>&1; then
        echo "$seq  $frames  contact L/R=$L/$R  FAILED  keypoints (see $log)"; continue
    fi

    if ! "$CONDA" run -n "$RETARGET_ENV" python contrack/retarget_crx_sharpa.py --h5 "$src" \
            --out "$OUT/$name-search-probe.h5" --overwrite "${urdf_args[@]}" \
            --hands "$HANDS" --arm-search >> "$log" 2>&1; then
        why=$(grep -oE 'first failed frames \[[^]]*\]|no acceptable first-frame candidate' "$log" | tail -1)
        echo "$seq  $frames  contact L/R=$L/$R  FAILED  ${why:-retarget error (see $log)}"; continue
    fi
    ts=$(grep -iE 'smallest --time-scale' "$log" | tail -1 | grep -oE '[0-9]+(\.[0-9]+)?$')
    line="$seq  $frames  contact L/R=$L/$R  OK  time-scale ${ts:-?}"

    if [ "$FINALIZE" = 1 ] && [ -n "${ts:-}" ]; then
        n=$(awk -v t="$ts" 'BEGIN { n = int(t); if (n < t) n++; if (n < 1) n = 1; print n }')
        tag=$([ "$HANDS" = right ] && echo "" || echo "-$HANDS")
        final="$D/$name-crx_sharpa$tag-search-slow$n.h5"
        if "$CONDA" run -n "$RETARGET_ENV" python contrack/retarget_crx_sharpa.py --h5 "$src" \
                --out "$final" --overwrite "${urdf_args[@]}" --hands "$HANDS" --arm-search \
                --time-scale "$n" >> "$log" 2>&1; then
            line="$line  -> $final"
        else
            line="$line  final conversion FAILED (see $log)"
        fi
    fi
    echo "$line"
done | tee -a "$OUT/summary.txt"
echo "Done. Summary: $OUT/summary.txt" >&2
