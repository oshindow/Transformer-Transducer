MAIN_ROOT=$PWD
KALDI_ROOT=$MAIN_ROOT/tools/kaldi

export PATH=$PWD/tt_utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

source $MAIN_ROOT/tools/venv/bin/activate

export PATH=$MAIN_ROOT/utils:$MAIN_ROOT/bin:$PATH
export OMP_NUM_THREADS=1

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8
