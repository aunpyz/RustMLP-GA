use super::rand::{thread_rng, Rng};
use std::cmp::Ordering;

mod function;
mod ga;

#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    weights: Vec<f64>,
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

    fn populate(
        input: usize,
        hidden_layer: &Vec<usize>,
        output: usize,
        population: usize,
    ) -> Vec<Self> {
        // weight from input to hidden layer & from hidden layer to output
        let mut weights: [usize; 2] = [0, 0];
        let mut networks: Vec<Self> = Vec::new();
        let cap = hidden_layer.len();

        weights[0] = input * hidden_layer[0];
        if cap == 1 {
            weights[1] = hidden_layer[0] * output;
        } else {
            for i in 0..cap {
                if i == cap - 1 {
                    weights[1] = hidden_layer[i] * output;
                } else {
                    weights[0] += hidden_layer[i] * hidden_layer[i + 1];
                }
            }
        }
        let total_size = weights[0] + weights[1];
        for _i in 0..population {
            networks.push(NeuralNetwork::new(total_size));
        }
        networks
    }

    fn weight_len(&self) -> usize {
        self.weights.len()
    }

    fn forward_pass(
        &self,
        data: &Vec<f64>,
        (input, hidden_layer, output): (usize, &Vec<usize>, usize),
    ) -> Vec<f64> {
        let next = hidden_layer[0];
        let mut outputs: Vec<f64> = Vec::with_capacity(output);
        let mut output_to_next: Vec<f64> = Vec::new();
        for i in 0..next {
            // skip output at index 0 & 1
            // but -2 when accessing self.weights
            let mut net: f64 = 0_f64;
            for j in 2..data.len() {
                let index = i * input + j - 2;
                net += data[j] * self.weights[index];
            }
            output_to_next.push(function::sigmoid(net));
        }

        {
            // layers after input node
            let mut base_index = input * hidden_layer[0];
            let cap = hidden_layer.len();
            let mut input_node = output_to_next;
            let mut output_to_next: Vec<f64> = Vec::new();

            if cap > 1 {
                for i in 0..cap {
                    let mut next;

                    if i == cap - 1 {
                        next = output;
                    } else {
                        next = hidden_layer[i + 1];
                    }

                    for j in 0..next {
                        let mut net: f64 = 0_f64;
                        for k in 0..input_node.len() {
                            let index = base_index + j * input_node.len() + k;
                            net += input_node[k] * self.weights[index];
                        }
                        output_to_next.push(function::sigmoid(net));
                    }

                    if i < cap - 1 {
                        base_index += hidden_layer[i] * hidden_layer[i + 1];
                        input_node = output_to_next.clone();
                        output_to_next.clear();
                    } else {
                        outputs.append(&mut output_to_next);
                    }
                }
            } else {
                let next = output;
                for i in 0..next {
                    let mut net: f64 = 0_f64;
                    for j in 0..input_node.len() {
                        let index = base_index + i * input_node.len() + j;
                        net += input_node[j] * self.weights[index];
                    }
                    output_to_next.push(function::sigmoid(net));
                }
                outputs.append(&mut output_to_next);
            }
        }

        for i in 0..output {
            let desired_output = data[i];
            outputs[i] = desired_output - outputs[i];
        }
        outputs
    }
}

pub fn cross_validation(
    (input, hidden_layer, output): (usize, Vec<usize>, usize),
    population: usize,
    validate_section: usize,
    epoch: usize,
    elitism_number: usize,
    min: f64,
    data: Vec<Vec<f64>>,
) {
    let k: i32 = thread_rng().gen_range(1, 5);
    let data_sections = function::split_section(data, validate_section);
    let master_chromosomes = NeuralNetwork::populate(input, &hidden_layer, output, population);
    for i in 0..validate_section {
        let mut chromosomes = ga::clone_population(&master_chromosomes);
        // total of epoch * population generations
        for _iter in 0..epoch {
            for j in 0..validate_section {
                // skip index i
                if j == i {
                    continue;
                }

                let data = &data_sections[j];
                // collection of fitness/ chromosome index pair
                let mut fitnesses: Vec<(f64, usize)> = Vec::with_capacity(population);
                for item in data.iter() {
                    for (index, chromosome) in chromosomes.iter().enumerate() {
                        let errors = chromosome.forward_pass(item, (input, &hidden_layer, output));
                        fitnesses.push((ga::fitness(errors, k), index));
                    }
                    fitnesses.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
                    assert_eq!(fitnesses.len(), population);
                    let next_gen = ga::elitism(&chromosomes, elitism_number, &fitnesses);
                    let mut mating_pool = ga::selection(&mut chromosomes, population, min, &fitnesses);
                    let p2 = ga::recombination(&mut mating_pool);
                    // mutation
                    // roll back
                    panic!("k: {}\n{:?}\n{:?}\n{:?}\nSTOP", k, fitnesses, mating_pool.len(), p2.len());
                }
            }
        }
    }
}
