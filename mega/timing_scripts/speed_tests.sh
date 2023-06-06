set -ex

cd "$(dirname "${BASH_SOURCE[0]}")/.."

GPU=0

bash timing_scripts/run_image.sh -1 $GPU
bash timing_scripts/run_image.sh 128 $GPU

bash timing_scripts/run_pathfinder.sh -1 $GPU
bash timing_scripts/run_pathfinder.sh 128 $GPU

bash timing_scripts/run_retrieval.sh -1 $GPU
bash timing_scripts/run_retrieval.sh 128 $GPU

bash timing_scripts/run_text.sh -1 $GPU
bash timing_scripts/run_text.sh 128 $GPU

bash timing_scripts/run_listops.sh -1 $GPU
bash timing_scripts/run_listops.sh 128 $GPU