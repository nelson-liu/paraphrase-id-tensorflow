#!/bin/bash
# This script is meant to be called by the "after_success" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.

set -e

if [[ "$COVERAGE" == "true" ]]; then
    # Ignore codecov failures as the codecov server is not
    # very reliable but we don't want travis to report a failure
    # in the github UI just because the coverage report failed to
    # be published.
    codecov --token=789eecab-ced6-4df9-b76b-36558d4e44c5 || echo "codecov upload failed"
fi
