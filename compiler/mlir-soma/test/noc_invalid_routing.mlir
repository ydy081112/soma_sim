module {
  noc.network @noc0 {
    topology = "mesh",
    dimensions = [8, 8],
    hop_latency = 2ns,
    routing = "static"
  }
}
