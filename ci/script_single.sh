#!/bin/bash

echo "[script_single]"

source activate sm2

if [ -n "$LOCALE_OVERRIDE" ]; then
    export LC_ALL="$LOCALE_OVERRIDE";
    echo "Setting LC_ALL to $LOCALE_OVERRIDE"

    pycmd='import pandas; print("pandas detected console encoding: %s" % pandas.get_option("display.encoding"))'
    python -c "$pycmd"
fi

if [ "$SLOW" ]; then
    TEST_ARGS="--only-slow"
fi



if [ "$DOC" ]; then
    echo "We are not running pytest as this is a doc-build"

elif [ "$COVERAGE" ]; then
    echo pytest -s -m "single" -r xXs --strict --cov=sm2 --cov-report xml:/tmp/cov-single.xml --junitxml=/tmp/single.xml $TEST_ARGS sm2
    pytest      -s -m "single" -r xXs --strict --cov=sm2 --cov-report xml:/tmp/cov-single.xml --junitxml=/tmp/single.xml $TEST_ARGS sm2

    echo pytest -s -r xXs --strict scripts
    pytest      -s -r xXs --strict scripts
else
    echo pytest -m "single" -r xXs --junitxml=/tmp/single.xml --strict $TEST_ARGS sm2
    pytest      -m "single" -r xXs --junitxml=/tmp/single.xml --strict $TEST_ARGS sm2 # TODO: doctest

fi

RET=$?

exit $RET
