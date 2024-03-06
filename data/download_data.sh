
# MoleculeNet + ZINC
wget http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip
unzip chem_dataset.zip
mv dataset molecule_datasets
rm chem_dataset.zip
rm -r molecule_datasets/*/processed

# for d in molecule_datasets/*/
# do
#     echo "$d"
#     ln -s $d ./
# done

# GEOM
wget https://dataverse.harvard.edu/api/access/datafile/4327252
mv 4327252 rdkit_folder.tar.gz
tar -xvf rdkit_folder.tar.gz
mv rdkit_folder GEOM/
rm rdkit_folder.tar.gz
