#!/bin/bash

echo "inside $0"

RET=0

if [ "$LINT" == true ]; then
    echo "Linting all files with limited rules"
    flake8 sm2
    if [ $? -ne "0" ]; then
        echo "Changed files failed linting using the required set of rules."
        echo "Additions and changes must conform to Python code style rules."
        RET=1
    fi

    # Run with --isolated to ignore config files, the files included here
    # pass _all_ flake8 checks
    echo "Linting known clean files with strict rules"
    flake8 --isolated \
        sm2/compat \
        sm2/datasets \
        sm2/distributions \
        sm2/formula \
        sm2/graphics \
        sm2/*.py \
        sm2/iolib/smpickle.py \
        sm2/iolib/tests/test_pickle.py \
        sm2/tools/linalg.py \
        sm2/tools/tests/test_linalg.py \
        sm2/tools/decorators.py \
        sm2/tools/tests/test_decorators.py \
        sm2/tsa/_bds.py \
        sm2/tsa/api.py \
        sm2/tsa/arima_process.py \
        sm2/tsa/autocov.py \
        sm2/tsa/stattools.py \
        sm2/tsa/tsatools.py \
        sm2/tsa/unit_root.py \
        sm2/tsa/wold.py \
        sm2/tsa/tests/test_stattools.py \
        sm2/tsa/kalmanf/ \
        sm2/tsa/regime_switching \
        sm2/tsa/vector_ar/hypothesis_test_results.py \
        setup.py
    if [ $? -ne "0" ]; then
        echo "Previously passing files failed linting."
        RET=1
    fi

    # Tests any new python files
    if [ -f $(git rev-parse --git-dir)/shallow ]; then
        # Unshallow only when required, i.e., on CI
        echo "Repository is shallow"
        git fetch --unshallow --quiet
    fi
    git config remote.origin.fetch "+refs/heads/*:refs/remotes/origin/*"
    git fetch origin --quiet
    NEW_FILES=$(git diff origin/master --name-status -u -- "*.py" | grep ^A | cut -c 3- | paste -sd " " -)
    if [ -n "$NEW_FILES" ]; then
        echo "Linting newly added files with strict rules"
        echo "New files: $NEW_FILES"
        flake8 --isolated $(eval echo $NEW_FILES)
        if [ $? -ne "0" ]; then
            echo "New files failed linting."
            RET=1
        fi
    else
        echo "No new files to lint"
    fi
fi

exit "$RET"
