import os

def create_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    print(f"Created {path}")

CHART_YAML = """apiVersion: v2
name: cad-ml-platform
description: A Helm chart for CAD ML Platform
type: application
version: 0.1.0
appVersion: "1.6.0"
"""

VALUES_YAML = """replicaCount: 1

image:
  repository: cad-ml-platform
  pullPolicy: IfNotPresent
  tag: "latest"

service:
  type: ClusterIP
  port: 8000

ingress:
  enabled: false
  className: ""
  annotations: {}
  hosts:
    - host: chart-example.local
      paths:
        - path: /
          pathType: ImplementationSpecific
  tls: []

resources: 
  limits:
    cpu: 1000m
    memory: 2Gi
  requests:
    cpu: 500m
    memory: 1Gi

autoscaling:
  enabled: false
  minReplicas: 1
  maxReplicas: 10
  targetCPUUtilizationPercentage: 80

env:
  VECTOR_STORE_BACKEND: "milvus"
  MILVUS_HOST: "milvus-standalone"
  MILVUS_PORT: "19530"
"""

DEPLOYMENT_YAML = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "cad-ml-platform.fullname" . }}
  labels:
    {{- include "cad-ml-platform.labels" . | nindent 4 }}
spec:
  {{- if not .Values.autoscaling.enabled }}
  replicas: {{ .Values.replicaCount }}
  {{- end }}
  selector:
    matchLabels:
      {{- include "cad-ml-platform.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      {{- with .Values.podAnnotations }}
      annotations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      labels:
        {{- include "cad-ml-platform.selectorLabels" . | nindent 8 }}
    spec:
      containers:
        - name: {{ .Chart.Name }}
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          ports:
            - name: http
              containerPort: {{ .Values.service.port }}
              protocol: TCP
          env:
            {{- range $key, $val := .Values.env }}
            - name: {{ $key }}
              value: {{ $val | quote }}
            {{- end }}
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
"""

SERVICE_YAML = """apiVersion: v1
kind: Service
metadata:
  name: {{ include "cad-ml-platform.fullname" . }}
  labels:
    {{- include "cad-ml-platform.labels" . | nindent 4 }}
spec:
  type: {{ .Values.service.type }}
  ports:
    - port: {{ .Values.service.port }}
      targetPort: http
      protocol: TCP
      name: http
  selector:
    {{- include "cad-ml-platform.selectorLabels" . | nindent 4 }}
"""

HELPERS_TPL = """{{/*
Expand the name of the chart.
*/}}
{{- define "cad-ml-platform.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "cad-ml-platform.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "cad-ml-platform.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "cad-ml-platform.labels" -}}
helm.sh/chart: {{ include "cad-ml-platform.chart" . }}
{{ include "cad-ml-platform.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "cad-ml-platform.selectorLabels" -}}
app.kubernetes.io/name: {{ include "cad-ml-platform.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
"""

def main():
    base_dir = "charts/cad-ml-platform"
    create_file(f"{base_dir}/Chart.yaml", CHART_YAML)
    create_file(f"{base_dir}/values.yaml", VALUES_YAML)
    create_file(f"{base_dir}/templates/deployment.yaml", DEPLOYMENT_YAML)
    create_file(f"{base_dir}/templates/service.yaml", SERVICE_YAML)
    create_file(f"{base_dir}/templates/_helpers.tpl", HELPERS_TPL)
    print("Helm chart generation complete.")

if __name__ == "__main__":
    main()
