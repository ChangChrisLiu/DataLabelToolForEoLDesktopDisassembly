#!/usr/bin/env bash
# One evaluation job: subset -> (blind) -> chat -> model -> score. (W4-B)
#
# Everything is an environment variable so the same file serves the Slurm
# templates, an interactive node and a laptop:
#
#   PACK        the vlmpack copy (required)              e.g. $SCRATCH/vlmpack_2026-11-02
#   VIEW        scan | oak1 | oak2 | rs | multi (required; multi = the action tasks: every
#               camera of a moment in one request, up to 8 images, or the marked
#               scanner image alone for V4, V6 multiple choice, V7 and V16)
#   BACKEND     gold | openai | vllm | transformers      (default vllm)
#   MODEL       model id: HF id / local path / API name  (required unless gold)
#   MODEL_TAG   directory name for the model             (default: MODEL with / -> _)
#   VARIANT     image | blind | overlay                  (default image; blind = image-ablated;
#                                                         overlay = GT masks painted on the frame, W4-C)
#   OVERLAYS    where the painted frames live / go       (default $PACK/overlays; the pack ships eval_v2's,
#                                                         missing ones are painted from the .masks.jsonl sidecar)
#   MASKS       a masks sidecar to paint from            (default: <truth>.masks.jsonl beside the subset)
#   SUBSET      eval | all | <ids file>                  (default eval: $PACK/eval_v2/$VIEW.jsonl;
#                                                         a path under eval_v1/ is refused)
#   RUNS        where runs go                            (default $PACK/runs)
#   TOOLS       the pack's tools directory               (default $PACK/tools)
#   IMAGE_ROOT  what the manifest paths are relative to  (default $PACK/images; see the pack README)
#   TRUTH       the unit's full export with sidecars     (default: $PACK/$VIEW.jsonl or $PACK/vlm_$VIEW.jsonl)
#   CHAT_FORMAT openai | qwen | internvl                 (default openai -- vlm_run reads all three)
#   ROI         none | view                              (default none)
#   MAX_SIDE    longest image side sent, pixels          (default: unchanged)
#   MAX_TOKENS  completion budget                        (default 2048: V1 answers run to ~1k tokens)
#   REPEAT      k answers per question                   (default 1; 3 for the closed-model repeat subset)
#   API_KEY_ENV / BASE_URL   openai backend: the NAME of the key variable, and the endpoint
#   RUN_ARGS    anything else for vlm_run.py             (e.g. "--tensor-parallel 2 --dtype bf16")
#   PYTHON      interpreter                              (default python)
#
# Output: $RUNS/$MODEL_TAG/$VARIANT/$VIEW/{truth/,chat.jsonl,chat_assets/,pred.jsonl,run.log,score/}
# (chat_assets/ holds the converter's copies: crops, scaled pictures, the marks drawn in red).
# Re-running resumes: vlm_run.py skips the ids already in pred.jsonl.
set -euo pipefail

: "${PACK:?set PACK to the vlmpack directory}"
: "${VIEW:?set VIEW to scan | oak1 | oak2 | rs | multi}"
case "$VIEW" in
  scan|oak1|oak2|rs|multi) ;;
  *) echo "VIEW must be scan, oak1, oak2, rs or multi (got $VIEW)" >&2; exit 2 ;;
esac
EVAL_VERSION=eval_v2
# the subset before the 2026-11-03 question redesign: its records are the old questions
for p in "${SUBSET:-}" "${TRUTH:-}"; do
  case "/$p/" in
    */eval_v1/*) echo "[run_eval] refused: $p is under eval_v1, the subset before the question redesign; use $EVAL_VERSION/" >&2; exit 2 ;;
  esac
done
BACKEND="${BACKEND:-vllm}"
MODEL="${MODEL:-}"
if [[ "$BACKEND" != "gold" && -z "$MODEL" ]]; then echo "set MODEL" >&2; exit 2; fi
MODEL_TAG="${MODEL_TAG:-${MODEL:-gold}}"
MODEL_TAG="${MODEL_TAG//\//_}"
VARIANT="${VARIANT:-image}"
SUBSET="${SUBSET:-eval}"
RUNS="${RUNS:-$PACK/runs}"
TOOLS="${TOOLS:-$PACK/tools}"
IMAGE_ROOT="${IMAGE_ROOT:-$PACK/images}"
CHAT_FORMAT="${CHAT_FORMAT:-openai}"
ROI="${ROI:-none}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
REPEAT="${REPEAT:-1}"
PYTHON="${PYTHON:-python}"
RUN_ARGS="${RUN_ARGS:-}"
# a multi request carries every camera of a moment (V3 / V12: of two moments): up to 8 images;
# a marked one (V4, V6 multiple choice, V7, V16) the scanner image alone
if [[ "$VIEW" == "multi" && "$BACKEND" == "vllm" && "$RUN_ARGS" != *--max-images* ]]; then
  RUN_ARGS="$RUN_ARGS --max-images 8"
fi

# a unit with no record is left out of the pack (manifest.json "units", export_vlmpack.py
# "empty_units"): its array task has nothing to run and ends cleanly
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${TRUTH:-}" && -f "$PACK/manifest.json" ]]; then
  UNITS=" $(PYTHON="$PYTHON" bash "$HERE/pack_units.sh" "$PACK") "
  if [[ "$UNITS" != *" $VIEW "* ]]; then
    echo "[run_eval] unit $VIEW not in this pack, skipped"
    exit 0
  fi
fi

# the view's full export (records + .images_manifest.jsonl + .oracle.jsonl + .summary.json)
if [[ -z "${TRUTH:-}" ]]; then
  for cand in "$PACK/data/vlm_$VIEW.jsonl" "$PACK/$VIEW.jsonl" "$PACK/vlm_$VIEW.jsonl" "$PACK/export/vlm_$VIEW.jsonl"; do
    if [[ -f "$cand" ]]; then TRUTH="$cand"; break; fi
  done
fi
: "${TRUTH:?no export found for $VIEW under $PACK; set TRUTH}"

OUT="$RUNS/$MODEL_TAG/$VARIANT/$VIEW"
mkdir -p "$OUT/truth" "$OUT/score"
RUN="$TOOLS/vlm_run"
echo "[run_eval] $BACKEND ${MODEL:-gold} / $VARIANT / $VIEW -> $OUT"

# 1. the records to run, as a truth file of their own (sidecars copied beside it)
SUB="$OUT/truth/$VIEW.jsonl"
case "$SUBSET" in
  eval) "$PYTHON" "$RUN/vlm_collect.py" subset --export "$TRUTH" --out "$SUB" \
          --records "$PACK/$EVAL_VERSION/$VIEW.jsonl" ;;
  all)  "$PYTHON" "$RUN/vlm_collect.py" subset --export "$TRUTH" --out "$SUB" --records "$TRUTH" ;;
  *)    "$PYTHON" "$RUN/vlm_collect.py" subset --export "$TRUTH" --out "$SUB" --ids "$SUBSET" ;;
esac

# 2. the input variant (the manifest the chat step reads, and what its paths are relative to)
RECORDS="$SUB"
CHAT_EXPORT="$SUB"
CHAT_ROOT="$IMAGE_ROOT"
if [[ "$VARIANT" == "blind" ]]; then
  RECORDS="$OUT/truth/blind.jsonl"
  "$PYTHON" "$RUN/vlm_collect.py" blind --records "$SUB" --out "$RECORDS"
elif [[ "$VARIANT" == "overlay" ]]; then
  # the GT-mask overlay: eligible records only (confirmed frames with masks, no V8);
  # the variant carries its own manifest, whose paths are the painted copies
  OVERLAYS="${OVERLAYS:-$PACK/overlays}"
  RECORDS="$OUT/truth/mask_overlay.jsonl"
  OVL=(render --export "$SUB" --image-root "$IMAGE_ROOT" --assets "$OVERLAYS" --out "$RECORDS")
  if [[ -n "${MASKS:-}" ]]; then OVL+=(--masks "$MASKS"); fi
  "$PYTHON" "$TOOLS/vlm_overlay.py" "${OVL[@]}"
  CHAT_EXPORT="$RECORDS"
  CHAT_ROOT="$OVERLAYS"
elif [[ "$VARIANT" != "image" ]]; then
  echo "VARIANT must be image, blind or overlay" >&2; exit 2
fi

# 3. chat requests (crops / scaled copies, if any, go under chat_assets/)
CHAT="$OUT/chat.jsonl"
TO_CHAT=(--export "$CHAT_EXPORT" --records "$RECORDS" --format "$CHAT_FORMAT" --image-root "$CHAT_ROOT"
         --out "$CHAT" --roi "$ROI" --assets "$OUT/chat_assets")
if [[ -n "${MAX_SIDE:-}" ]]; then TO_CHAT+=(--max-side "$MAX_SIDE"); fi
"$PYTHON" "$TOOLS/vlm_to_chat.py" "${TO_CHAT[@]}"

# 4. the model
PRED="$OUT/pred.jsonl"
ARGS=(--backend "$BACKEND" --chat "$CHAT" --out "$PRED" --image-root "$CHAT_ROOT"
      --max-tokens "$MAX_TOKENS" --repeat "$REPEAT" --log "$OUT/run.log")
case "$BACKEND" in
  gold)   ARGS+=(--truth "$SUB") ;;
  openai) ARGS+=(--model "$MODEL" --base-url "${BASE_URL:?set BASE_URL}"
                 --api-key-env "${API_KEY_ENV:?set API_KEY_ENV to the NAME of the key variable}") ;;
  *)      ARGS+=(--model "$MODEL") ;;
esac
# shellcheck disable=SC2086
"$PYTHON" "$RUN/vlm_run.py" "${ARGS[@]}" $RUN_ARGS || status=$?
status="${status:-0}"
if [[ "$status" == "3" ]]; then echo "[run_eval] some requests failed; re-run to retry them" >&2; fi
if [[ "$status" != "0" && "$status" != "3" ]]; then exit "$status"; fi

# 5. scores: the file as it is (raw_text = repeat 1) and, when repeated, each repeat.
#    The truth is always the plain subset: the overlay variant answers fewer records
#    (its summary says which were left out), and the unanswered ones count as parse failures.
"$PYTHON" "$TOOLS/vlm_score.py" --truth "$SUB" --pred "$PRED" --chat "$CHAT" \
  --out "$OUT/score" --name "$MODEL_TAG"
if [[ "$REPEAT" -gt 1 ]]; then
  "$PYTHON" "$RUN/vlm_collect.py" split --pred "$PRED" --out-dir "$OUT"
  for i in $(seq 1 "$REPEAT"); do
    "$PYTHON" "$TOOLS/vlm_score.py" --truth "$SUB" --pred "$OUT/pred.r$i.jsonl" --chat "$CHAT" \
      --out "$OUT/score" --name "$MODEL_TAG.r$i"
  done
fi
echo "[run_eval] scores -> $OUT/score/$MODEL_TAG.score.md"
exit "$status"
