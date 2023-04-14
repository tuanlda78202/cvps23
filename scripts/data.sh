# Download data and unzip 
gdown link/drive/of/data -O data/data.zip
unzip data.zip > /dev/null
rm -rf data/data.zip

# Tree
apt-get -q install tree
tree data

# TO-DO LIST
# Export ~100k sample for project 
# Split train, val, test 