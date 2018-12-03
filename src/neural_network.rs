use super::rand::{thread_rng, Rng};

#[derive(Debug)]
pub struct NeuralNetwork {
    weights: Vec<f64>,
}

#[derive(Debug)]
enum NeuronType {
    input(usize),
    hidden_layer(Vec<usize>),
    output(usize),
}

impl NeuralNetwork {
    // number of inputs, hidden layers, outputs are defined elsewhere
    fn new(size: usize) -> Self {
        let mut weights: Vec<f64> = Vec::new();
        let mut rng = thread_rng();
        for _i in 0..size {
            weights.push(rng.gen_range(-1_f64, 1_f64));
        }

        NeuralNetwork { weights }
    }

    pub fn populate(
        input: usize,
        hidden_layer: Vec<usize>,
        output: usize,
        population: usize,
    ) -> Vec<Self> {
        let mut hidden_layers = 0;
        let mut networks: Vec<Self> = Vec::new();
        for layer in &hidden_layer {
            hidden_layers += layer;
        }
        let total_size = input + output + hidden_layers;
        for _i in 0..population {
            networks.push(NeuralNetwork::new(total_size));
        }
        networks
    }
}
