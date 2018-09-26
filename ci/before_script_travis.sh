#!/bin/bash
# copied from pandas

echo "inside $0"

# Fix for headless TravisCI
#   https://stackoverflow.com/questions/35403127
if [ "${TRAVIS_OS_NAME}" == "linux" ]; then
   sh -e /etc/init.d/xvfb start
   sleep 3
fi

# Never fail because bad things happened here.
true
