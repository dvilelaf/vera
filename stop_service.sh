#!/bin/bash

# ------------------------------------------------------------------------------
#
#   Copyright 2024 David Vilela Freire
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------

REPO_PATH=$PWD
VERA_DB=$REPO_PATH/vera/abci_build/persistent_data/logs/vera.db

# poetry run autonomy deploy stop --build-dir vera/abci_build; cd ..
docker container stop vera_abci_0 vera_tm_0

# Backup db
if test -e $VERA_DB; then
  echo "Creating database backup"
  cp $VERA_DB $REPO_PATH
fi