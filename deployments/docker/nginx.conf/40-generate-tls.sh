#!/bin/sh
set -eu

tls_src_dir="/etc/nginx/ssl"
tls_work_dir="/tmp/nginx-ssl"
tls_cert_path="${tls_work_dir}/tls.crt"
tls_key_path="${tls_work_dir}/tls.key"
tls_days="${CAD_ML_TLS_DAYS:-365}"
tls_cn="${CAD_ML_TLS_CN:-localhost}"
tls_san="${CAD_ML_TLS_SAN:-DNS:localhost,IP:127.0.0.1,DNS:cad-ml-nginx}"

mkdir -p "${tls_work_dir}"

if [ -f "${tls_src_dir}/tls.crt" ] && [ -f "${tls_src_dir}/tls.key" ]; then
    cp "${tls_src_dir}/tls.crt" "${tls_cert_path}"
    cp "${tls_src_dir}/tls.key" "${tls_key_path}"
    chmod 600 "${tls_key_path}"
    echo "nginx tls: using mounted certificate from ${tls_src_dir}"
    exit 0
fi

openssl req \
    -x509 \
    -nodes \
    -newkey rsa:2048 \
    -keyout "${tls_key_path}" \
    -out "${tls_cert_path}" \
    -days "${tls_days}" \
    -subj "/CN=${tls_cn}" \
    -addext "subjectAltName = ${tls_san}" \
    >/dev/null 2>&1

chmod 600 "${tls_key_path}"
echo "nginx tls: generated self-signed certificate at ${tls_cert_path}"
