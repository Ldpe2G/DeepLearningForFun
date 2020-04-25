basepath=$(cd `dirname $0`/; pwd)

BUILD_DIR=${basepath}/build

rm -rf ${BUILD_DIR}
if [[ ! -d ${BUILD_DIR} ]]; then
    mkdir -p ${BUILD_DIR}
fi

cd ${BUILD_DIR}
cmake ..
make -j4

python3 ${basepath}/testData/test.py

./read_nd test-0000.params
