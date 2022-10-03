module github.com/seldonio/seldon-core/executor

go 1.16

require (
	contrib.go.opencensus.io/exporter/ocagent v0.4.12 // indirect
	contrib.go.opencensus.io/exporter/prometheus v0.1.0 // indirect
	github.com/Azure/azure-sdk-for-go v30.1.0+incompatible // indirect
	github.com/Azure/go-autorest/autorest/to v0.2.0 // indirect
	github.com/Azure/go-autorest/autorest/validation v0.1.0 // indirect
	github.com/cloudevents/sdk-go v2.12.0+incompatible
	github.com/confluentinc/confluent-kafka-go v1.4.2
	github.com/fortytw2/leaktest v1.3.0 // indirect
	github.com/ghodss/yaml v1.0.0
	github.com/go-logr/logr v0.4.0
	github.com/golang/protobuf v1.5.2
	github.com/google/uuid v1.2.0
	github.com/gorilla/mux v1.8.0
	github.com/grpc-ecosystem/go-grpc-middleware v1.2.1
	github.com/kelseyhightower/envconfig v1.4.0 // indirect
	github.com/lightstep/tracecontext.go v0.0.0-20181129014701-1757c391b1ac // indirect
	github.com/nats-io/nats-server/v2 v2.1.2 // indirect
	github.com/onsi/gomega v1.14.0
	github.com/opentracing/opentracing-go v1.2.0
	github.com/pkg/errors v0.9.1
	github.com/prometheus/client_golang v1.11.0
	github.com/prometheus/common v0.26.0
	github.com/seldonio/seldon-core/operator v0.0.0-00010101000000-000000000000
	github.com/tensorflow/tensorflow/tensorflow/go/core v0.0.0-00010101000000-000000000000
	github.com/uber/jaeger-client-go v2.25.0+incompatible
	github.com/uber/jaeger-lib v2.2.0+incompatible // indirect
	github.com/valyala/bytebufferpool v1.0.0 // indirect
	go.uber.org/automaxprocs v1.4.0
	go.uber.org/zap v1.19.0
	golang.org/x/xerrors v0.0.0-20200804184101-5ec99f83aff1
	google.golang.org/grpc v1.37.0
	gotest.tools v2.2.0+incompatible
	k8s.io/api v0.21.3
	pack.ag/amqp v0.11.0 // indirect
	sigs.k8s.io/controller-runtime v0.9.6
)

replace github.com/tensorflow/tensorflow/tensorflow/go/core => ./proto/tensorflow/core

replace github.com/seldonio/seldon-core/operator => ./_operator

replace k8s.io/client-go => k8s.io/client-go v0.21.3

replace github.com/codahale/hdrhistogram => github.com/HdrHistogram/hdrhistogram-go v1.1.2

exclude github.com/go-logr/logr v1.0.0
