#!/bin/sh

set -e

(
mkdir original_data
cd original_data

wget https://ufal.mff.cuni.cz/~mnovak/files/crac26/unc-gold-train.zip
wget https://ufal.mff.cuni.cz/~mnovak/files/crac26/unc-gold-minidev.zip

unzip unc-gold-train.zip
unzip unc-gold-minidev.zip
)

for tb in ca_ancora cs_pcedt cs_pdt cs_pdtsc cu_proiel es_ancora grc_proiel hu_korkor hu_szegedkoref pl_pcc tr_itcc; do
  mv original_data/$tb-*.conllu .
done

rm -rf original_data

echo All done
