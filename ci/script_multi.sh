#!/bin/bash -e

echo "[script multi]"

source activate sm2

if [ -n "$LOCALE_OVERRIDE" ]; then
    export LC_ALL="$LOCALE_OVERRIDE";
    echo "Setting LC_ALL to $LOCALE_OVERRIDE"

    pycmd='import pandas; print("pandas detected console encoding: %s" % pandas.get_option("display.encoding"))'
    python -c "$pycmd"
fi


# Workaround for pytest-xdist flaky collection order
# https://github.com/pytest-dev/pytest/issues/920
# https://github.com/pytest-dev/pytest/issues/1075
export PYTHONHASHSEED=$(python -c 'import random; print(random.randint(1, 4294967295))')
echo PYTHONHASHSEED=$PYTHONHASHSEED

if [ "$DOC" ]; then
    echo "We are not running pytest as this is a doc-build"

elif [ "$COVERAGE" ]; then
    echo pytest -s -n 2 -m "not single" --cov=sm2 --cov-report xml:/tmp/cov-multiple.xml --junitxml=/tmp/multiple.xml --strict $TEST_ARGS sm2
    pytest      -s -n 2 -m "not single" --cov=sm2 --cov-report xml:/tmp/cov-multiple.xml --junitxml=/tmp/multiple.xml --strict $TEST_ARGS sm2

elif [ "$SLOW" ]; then
    TEST_ARGS="--only-slow"
    echo pytest -r xX -m "not single and slow" -v --junitxml=/tmp/multiple.xml --strict $TEST_ARGS sm2
    pytest -r xX -m "not single and slow" -v --junitxml=/tmp/multiple.xml --strict $TEST_ARGS sm2

else
    echo pytest -n 2 -r xX -m "not single" --junitxml=/tmp/multiple.xml --strict $TEST_ARGS sm2
    pytest      -n 2 -r xX -m "not single" --junitxml=/tmp/multiple.xml --strict $TEST_ARGS sm2 # TODO: doctest

fi

RET="$?"

exit "$RET"