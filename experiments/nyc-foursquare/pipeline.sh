#!/bin/bash

# Build data representations using NYCFoursquareData data source.
# Automatically builds STEP and Applications-specific representations.
python $STLARM_BASE_DIR/stlarm/scripts/build_data.py \
    --verbose \
    --data-source-cls NYCFoursquareData \
    --name "nyc_foursquare" \
    --working-dir . \
    --time-granularity period \
    --max-hours-diff 2 \
    --max-distance-km 2 \
    >  build_data.log \
    2>&1

mkdir -p mining

# Mine rules
java -jar $AMIE_JAR_PATH \
    -const \
    -oute -verbose \
    -bias lazy -full \
    -iexr "<hasCheckin>,<before>,<withinRadius>,<withinTimeWindow>" \
    -rl 2 \
    -pm support \
    -minis 500 -mins 50 \
    -maxad 4 \
    resources/converted_nyc_foursquare_period_2_2_None_filtered_triples.nt \
    > mining/mined_rules.tsv \
    2> mining/amie_mining.log

# Build metarules mapping
java -cp $AMIE_JAR_PATH \
    amie.rules.eval.MetaruleBuilder \
    mining/mined_rules.tsv \
    > mining/metarules.tsv
