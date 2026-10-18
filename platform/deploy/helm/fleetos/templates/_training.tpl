{{/*
Training / model-gate env vars shared by the api and worker containers.
The api submits training + eval WorkflowTemplates and gates deploys; the worker
runs the Temporal training pipeline. Both talk to the Argo Server REST API.
*/}}
{{- define "fleetos.trainingEnv" -}}
- name: ARGO_SERVER_URL
  value: {{ .Values.training.argo.serverURL | default "" | quote }}
- name: ARGO_NAMESPACE
  value: {{ .Values.training.argo.namespace | default "argo" | quote }}
- name: ARGO_INSECURE_SKIP_VERIFY
  value: {{ .Values.training.argo.insecureSkipVerify | default false | toString | quote }}
{{- with .Values.training.argo.tokenSecret }}
- name: ARGO_TOKEN
  valueFrom:
    secretKeyRef:
      name: {{ . }}
      key: token
{{- end }}
- name: MODELS_BUCKET
  value: {{ .Values.training.modelsBucket | default "fleetos-models" | quote }}
- name: TELEMETRY_BUCKET
  value: {{ .Values.training.telemetryBucket | default "fleetos-telemetry" | quote }}
- name: MODEL_MIN_EVAL_PASS_RATE
  value: {{ .Values.training.minEvalPassRate | default "0.5" | toString | quote }}
- name: KUBEFLOW_ENABLED
  value: {{ .Values.kubeflow.enabled | toString | quote }}
{{- if .Values.kubeflow.enabled }}
- name: KUBEFLOW_NAMESPACE
  value: {{ .Values.kubeflow.namespace | quote }}
- name: KUBEFLOW_RUNTIME
  value: {{ .Values.kubeflow.runtime | quote }}
- name: TRAINING_IMAGE
  value: "{{ .Values.kubeflow.image.repository }}:{{ .Values.kubeflow.image.tag }}"
- name: HPO_ALGORITHM
  value: {{ .Values.kubeflow.hpo.algorithm | quote }}
- name: HPO_MAX_TRIALS
  value: {{ .Values.kubeflow.hpo.maxTrials | toString | quote }}
- name: HPO_PARALLEL_TRIALS
  value: {{ .Values.kubeflow.hpo.parallelTrials | toString | quote }}
- name: HPO_TRIAL_GRADIENT_STEPS
  value: {{ .Values.kubeflow.hpo.trialGradientSteps | toString | quote }}
- name: HPO_EVAL_EPISODES
  value: {{ .Values.kubeflow.hpo.evalEpisodes | toString | quote }}
- name: KUBEFLOW_TRAINING_SECRET
  value: {{ .Values.kubeflow.trainingSecret | default "fleetos-training" | quote }}
{{- end }}
{{- end -}}
