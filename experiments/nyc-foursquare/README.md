# NYC Foursquare Experiment

This folder contains everything related to the experiment using NYC Foursquare data from [[1]](#yang).

## Folders structure

- `data_resources`: auxiliary extra semantic data for the NYC Foursquare. Execute the notebook in this folder before `pipeline.sh`. Data was provided by authors of [[2]](#petry), and was previously generated using code from [lucaspetry/trajectory-data](https://github.com/lucaspetry/trajectory-data).
- `pipeline.sh`: script for building data using `stsarm/scripts/build_data.py`, mining rules using AMIE 3, and building metarules.
- `mining`: mined rules and log.
- `cache`: folder with owlready2's persistent ontology.
- `resources`: folder with ontologies in triples format.
- `analyses`: notebooks with data and rules analyses.

<a name="yang">\[1\]</a>: Dingqi Yang, Daqing Zhang, V. W. Zheng, and Zhiyong Yu, “Modeling User Activity Preference by Leveraging User Spatial Temporal Characteristics in LBSNs,” IEEE Transactions on Systems, Man, and Cybernetics: Systems, vol. 45, no. 1, pp. 129–142, Jan. 2015, doi: 10/gf4vf2.


<a name="petry">\[2\]</a>: L. M. Petry, C. A. Ferrero, L. O. Alvares, C. Renso, and V. Bogorny, “Towards semantic-aware multiple-aspect trajectory similarity measuring,” Transactions in GIS, Jun. 2019, doi: 10/gf4hzb.


