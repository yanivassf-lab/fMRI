#!/bin/bash

# Define the correct source directory
SRC_DIR="/mnt/Amir_data/To Refael"

# Define the destination directories
TRAIN_DIR="/mnt/my_passport/R_output/raw_data_train"
TEST_DIR="/mnt/my_passport/R_output/raw_data_test"

# Create directories
mkdir -p "$TRAIN_DIR"
mkdir -p "$TEST_DIR"

# Train Set (114 Subjects)
declare -A TRAIN_MAP=(
    ["sub-S078478"]="sub-YA1293"
    ["sub-S415668"]="sub-YA1425"
    ["sub-S074222"]="sub-YA1301"
    ["sub-S899172"]="sub-YA1315"
    ["sub-S878485"]="sub-FNCL49"
    ["sub-S407235"]="sub-YA1393"
    ["sub-S950869"]="sub-YA1381"
    ["sub-S603931"]="sub-YA1299"
    ["sub-S448341"]="sub-YA1356"
    ["sub-S288220"]="sub-YA1423"
    ["sub-S344637"]="sub-YA1344"
    ["sub-S119155"]="sub-FNCL52"
    ["sub-S222954"]="sub-YA1394"
    ["sub-S949448"]="sub-YA1312"
    ["sub-S399661"]="sub-YA1414"
    ["sub-S168832"]="sub-YA1291"
    ["sub-S137901"]="sub-YA1323"
    ["sub-S062567"]="sub-YA1332"
    ["sub-S234123"]="sub-YA1464"
    ["sub-S167107"]="sub-YA1437"
    ["sub-S783149"]="sub-YA1329"
    ["sub-S606431"]="sub-YA1456"
    ["sub-S046716"]="sub-YA1294"
    ["sub-S088278"]="sub-YA1424"
    ["sub-S073454"]="sub-YA1316"
    ["sub-S706987"]="sub-YA1383"
    ["sub-S495362"]="sub-YA1348"
    ["sub-S184540"]="sub-YA1432"
    ["sub-S178484"]="sub-CHSL38"
    ["sub-S038630"]="sub-YA1297"
    ["sub-S904429"]="sub-YA1347"
    ["sub-S126259"]="sub-YA1309"
    ["sub-S778746"]="sub-FTNL05"
    ["sub-S789653"]="sub-YA1319"
    ["sub-S570113"]="sub-FTNC11"
    ["sub-S168705"]="sub-YA1363"
    ["sub-S521615"]="sub-FNCL02"
    ["sub-S430219"]="sub-MRTC18"
    ["sub-S316172"]="sub-YA1349"
    ["sub-S105498"]="sub-YA1388"
    ["sub-S730729"]="sub-YA1441"
    ["sub-S930980"]="sub-YA1403"
    ["sub-S716676"]="sub-YA1217"
    ["sub-S708740"]="sub-YA1339"
    ["sub-S657834"]="sub-YA1358"
    ["sub-S022768"]="sub-YA1357"
    ["sub-S479469"]="sub-YA1442"
    ["sub-S741055"]="sub-YA1310"
    ["sub-S668359"]="sub-YA1308"
    ["sub-S006143"]="sub-FTNC09"
    ["sub-S121576"]="sub-YA1448"
    ["sub-S295212"]="sub-YA1298"
    ["sub-S176993"]="sub-YA1334"
    ["sub-S283737"]="sub-YA1351"
    ["sub-S690045"]="sub-YA1355"
    ["sub-S298347"]="sub-BB01856"
    ["sub-S895733"]="sub-YA0579"
    ["sub-S241699"]="sub-YA1387"
    ["sub-S857471"]="sub-AGN08"
    ["sub-S019525"]="sub-YA1460"
    ["sub-S651922"]="sub-YA1420"
    ["sub-S761974"]="sub-YA1405"
    ["sub-S693000"]="sub-YA1419"
    ["sub-S524533"]="sub-YA1478"
    ["sub-S970316"]="sub-YA1411"
    ["sub-S649139"]="sub-YA1375"
    ["sub-S393368"]="sub-YA1452"
    ["sub-S566709"]="sub-YA1453"
    ["sub-S147749"]="sub-FNCL54"
    ["sub-S035513"]="sub-AGN11"
    ["sub-S630685"]="sub-YA1459"
    ["sub-S900609"]="sub-YA1461"
    ["sub-S283274"]="sub-YA1402"
    ["sub-S077465"]="sub-YA1434"
    ["sub-S464027"]="sub-YA1302"
    ["sub-S975494"]="sub-YA1343"
    ["sub-S836546"]="sub-GYML41"
    ["sub-S372165"]="sub-YA1318"
    ["sub-S747311"]="sub-FNCL51"
    ["sub-S840835"]="sub-YA1290"
    ["sub-S279520"]="sub-YA1410"
    ["sub-S017928"]="sub-YA1324"
    ["sub-S921428"]="sub-FNCL43"
    ["sub-S184816"]="sub-YA1341"
    ["sub-S377634"]="sub-FNCL48"
    ["sub-S628099"]="sub-YA1445"
    ["sub-S099846"]="sub-GYML38"
    ["sub-S779270"]="sub-YA1321"
    ["sub-S389860"]="sub-YA0474"
    ["sub-S847695"]="sub-YA1446"
    ["sub-S848037"]="sub-YA1476"
    ["sub-S986319"]="sub-FNCL60"
    ["sub-S640179"]="sub-BJJL180"
    ["sub-S962918"]="sub-YA1279"
    ["sub-S058477"]="sub-YA1421"
    ["sub-S216298"]="sub-YA1454"
    ["sub-S980319"]="sub-YA1314"
    ["sub-S572757"]="sub-YA1408"
    ["sub-S265961"]="sub-YA1322"
    ["sub-S305198"]="sub-YA1335"
    ["sub-S680340"]="sub-YA0879"
    ["sub-S579837"]="sub-YA1289"
    ["sub-S451252"]="sub-YA1305"
    ["sub-S832283"]="sub-YA1463"
    ["sub-S472660"]="sub-YA1331"
    ["sub-S234652"]="sub-YA1377"
    ["sub-S847190"]="sub-YA1369"
    ["sub-S739710"]="sub-YA1370"
    ["sub-S670463"]="sub-YA1296"
    ["sub-S867794"]="sub-YA1345"
    ["sub-S081375"]="sub-YA1346"
    ["sub-S881149"]="sub-BJJL153"
    ["sub-S109414"]="sub-YA1426"
    ["sub-S545263"]="sub-YA1390"
    ["sub-S955070"]="sub-YA1473"
)

# Test Set (50 Subjects)
declare -A TEST_MAP=(
    ["sub-S016284"]="sub-YA1480"
    ["sub-S911780"]="sub-MRTC08"
    ["sub-S614041"]="sub-CLML53"
    ["sub-S352790"]="sub-YA1367"
    ["sub-S850150"]="sub-YA1466"
    ["sub-S435149"]="sub-YA1380"
    ["sub-S694624"]="sub-GYML33"
    ["sub-S018537"]="sub-BB00932"
    ["sub-S095575"]="sub-YA1376"
    ["sub-S795045"]="sub-YA1427"
    ["sub-S051813"]="sub-YA1389"
    ["sub-S636287"]="sub-YA1327"
    ["sub-S516551"]="sub-YA1365"
    ["sub-S756277"]="sub-YA1306"
    ["sub-S117502"]="sub-YA1340"
    ["sub-S707850"]="sub-PYTL19"
    ["sub-S298159"]="sub-YA1336"
    ["sub-S440227"]="sub-YA1443"
    ["sub-S970464"]="sub-FNCL47"
    ["sub-S516845"]="sub-MRTC11"
    ["sub-S256643"]="sub-YA1311"
    ["sub-S617362"]="sub-YA1366"
    ["sub-S919556"]="sub-YA1307"
    ["sub-S106587"]="sub-YA1373"
    ["sub-S018839"]="sub-YA1471"
    ["sub-S992563"]="sub-YA1300"
    ["sub-S264484"]="sub-YA1368"
    ["sub-S148331"]="sub-YA1469"
    ["sub-S839850"]="sub-YA1430"
    ["sub-S597344"]="sub-YA1458"
    ["sub-S589990"]="sub-YA1359"
    ["sub-S240481"]="sub-FNCL50"
    ["sub-S469863"]="sub-YA1364"
    ["sub-S342935"]="sub-MRTC20"
    ["sub-S228186"]="sub-YA1431"
    ["sub-S332560"]="sub-YA1391"
    ["sub-S435529"]="sub-CHSC04"
    ["sub-S711330"]="sub-YA1378"
    ["sub-S763537"]="sub-YA0473"
    ["sub-S325680"]="sub-YA1295"
    ["sub-S163020"]="sub-YA1451"
    ["sub-S442685"]="sub-YA1325"
    ["sub-S405091"]="sub-FNCL46"
    ["sub-S077330"]="sub-YA1362"
    ["sub-S047601"]="sub-YA1303"
    ["sub-S287963"]="sub-YA1398"
    ["sub-S773032"]="sub-YA1361"
    ["sub-S110307"]="sub-YA1304"
    ["sub-S158100"]="sub-YA1371"
    ["sub-S583614"]="sub-YA1418"
    ["sub-S763263"]="sub-YA1482"
)

# Prevent loop from running with literal asterisks
shopt -s nullglob


# Reusable function to map and link
create_links() {
    local -n map=$1
    local dest_dir=$2

    echo "Processing to $dest_dir..."

    for SUB in "${!map[@]}"; do
        MAPPED_SUB="${map[$SUB]}"

        # Explicitly target ONLY the required file formats (now including JSON for bold and mask)
        for FILEPATH in "$SRC_DIR"/"$SUB"_*movement*space*preproc_bold.nii.gz \
                        "$SRC_DIR"/"$SUB"_*movement*space*preproc_bold.json \
                        "$SRC_DIR"/"$SUB"_*movement*space*brain_mask.nii.gz \
                        "$SRC_DIR"/"$SUB"_*movement*space*brain_mask.json \
                        "$SRC_DIR"/"$SUB"_*movement*confounds_timeseries.json \
                        "$SRC_DIR"/"$SUB"_*movement*confounds_timeseries.tsv; do

            if [ -f "$FILEPATH" ]; then
                FILENAME=$(basename "$FILEPATH")

                # 1. Replace the original subject ID with the mapped ID
                NEW_FILENAME="${FILENAME/$SUB/$MAPPED_SUB}"

                # 2. Remove 'music' while keeping 'movement' and the run number intact
                NEW_FILENAME="${NEW_FILENAME//musicmovement/movement}"

                # Create the symbolic link
                ln -s "$FILEPATH" "$dest_dir/$NEW_FILENAME"
            fi
        done
    done
}

# Create the specific symlinks
create_links TRAIN_MAP "$TRAIN_DIR"
create_links TEST_MAP "$TEST_DIR"

shopt -u nullglob
echo "Done! The raw data has been split into train and test directories."
