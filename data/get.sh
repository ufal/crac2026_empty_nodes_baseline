#!/bin/sh

set -e

if ! command -v udapy >/dev/null 2>&1; then
  echo "The 'udapy' command was not found. You can install it for example by creating" >&2
  echo "a Python virtual environment, activating it, and running 'pip install udapi'." >&2
  exit 1
fi

(
mkdir original_data
cd original_data

wget https://ufal.mff.cuni.cz/~mnovak/files/crac26/unc-gold-train.zip
wget https://ufal.mff.cuni.cz/~mnovak/files/crac26/unc-gold-minidev.zip

unzip unc-gold-train.zip
unzip unc-gold-minidev.zip
)

for tb in ca_ancora cs_pcedt cs_pdt cs_pdtsc cu_proiel es_ancora grc_proiel hu_korkor hu_szegedkoref pl_pcc tr_itcc; do
  udapy -s corefud.SingleParent <original_data/$tb-corefud-train.conllu >$tb-corefud-train.conllu
  mv original_data/$tb-corefud-minidev.conllu .
done

rm -rf original_data

echo All done
