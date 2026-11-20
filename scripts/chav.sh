#!/usr/bin/env bash
# Chav helper (Git Bash on Lucas's laptop; needs the priism checkout, its venv and the R2 profile).
# Chav is a remote 12-CPU / RTX 4050 machine running the `relay` job agent of the priism project
# (reference: C:\Users\lucas\Projects\priism-cleave\docs\chav-agent.md). Only the commands declared
# in relay.toml can run there.
#   chav.sh status [n]                   agents and recent acc jobs
#   chav.sh log ID [lines]               end of a job's log
#   chav.sh push [BRANCH]                ship a committed branch (default main), wait for the install,
#                                        run the tests on the CPU lane
#   chav.sh job MAXMIN RESULTS KIND ...  submit one CPU job (RESULTS '-' for none), wait at most ~8 min,
#                                        print the end of its log, pull its results into data/processed/
#                                        Prints "STILL RUNNING id" if it is not done.
#   chav.sh wait ID [RESULTS]            wait again for a job that was still running
# Every wait is bounded: no command of this script can hang.
cd "/c/Users/lucas/Projects/Allergen-Chip-Challenge" || exit 1
export PYTHONPATH='C:\Users\lucas\Projects\priism-cleave\src' RELAY_STORE='s3://priism-jobs/queue' \
  AWS_PROFILE='priism-r2' AWS_DEFAULT_REGION='auto' PYTHONIOENCODING='utf-8' \
  AWS_ENDPOINT_URL='https://f098c4b3a591f78c07382e384992dbd5.r2.cloudflarestorage.com'
PY="/c/Users/lucas/Projects/priism/.venv/Scripts/python.exe"
relay() { "$PY" -m relay.cli "$@" 2>&1; }

state_of() { echo "$1" | awk '{print $1}'; }

wait_for() {  # id, polls, seconds between polls
  for _ in $(seq 1 "$2"); do
    line=$(relay jobs status | rg -F "$1" | head -1)
    case "$(state_of "$line")" in done|failed|cancelled) break ;; esac
    sleep "$3"
  done
  echo "$line" | cut -c1-220
}

finish() {  # id, results path or '-'
  line=$(wait_for "$1" "${POLLS:-24}" 20)
  echo "$line"
  relay jobs log "$1" | rg -v '^\s*$' | cut -c1-260 | tail -"${TAIL:-30}"
  case "$(state_of "$line")" in
    done|failed|cancelled)
      if [ "$2" != "-" ]; then
        mkdir -p data/processed && (cd data/processed && relay jobs pull "$1" | tail -1)
      fi ;;
    *) echo "STILL RUNNING $1" ;;
  esac
}

case "$1" in
  status)
    relay jobs status | rg '^agent|acc\.' | cut -c1-200 | tail -"${2:-12}"
    ;;
  log)
    relay jobs log "$2" | rg -v '^\s*$' | cut -c1-260 | tail -"${3:-30}"
    ;;
  push)
    U=$(relay push acc --repo 'C:\Users\lucas\Projects\Allergen-Chip-Challenge' --branch "${2:-main}" | tail -1)
    echo "update $U: $(wait_for "$U" 30 20)"
    relay jobs log "$U" | rg -v '^\s*$' | tail -5
    T=$(relay jobs --max-minutes 20 submit --cpu acc/pytest | tail -1)
    echo "pytest: $(wait_for "$T" 24 20)"
    relay jobs log "$T" | rg -v '^\s*$' | tail -4
    ;;
  job)
    MAX=$2; RES=$3; shift 3
    if [ "$RES" = "-" ]; then ID=$(relay jobs --max-minutes "$MAX" submit --cpu "$@" | tail -1)
    else ID=$(relay jobs --max-minutes "$MAX" submit --cpu --results "$RES" "$@" | tail -1); fi
    echo "job: $ID"
    finish "$ID" "$RES"
    ;;
  wait)
    finish "$2" "${3:--}"
    ;;
  *) echo "usage: chav.sh status|log|push|job|wait" ;;
esac
