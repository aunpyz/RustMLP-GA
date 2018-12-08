use super::rand::{thread_rng, Rng};
use std::cmp::Ordering;
use std::fmt;
use std::error::Error;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

mod function;
mod ga;

#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    weights: Vec<f64>,
}

impl fmt::Display for NeuralNetwork {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut display = String::new();
        display.push_str(&format!("Neural Network\nweight: "));
        for i in 0..self.weights.len() {
            display.push_str(&format!("{}, ", self.weights[i]));
        }
        write!(f, "{}\n", display)
    }
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
        train: bool,
    ) -> (Vec<f64>, Vec<f64>) {
        let next = hidden_layer[0];
        let mut outputs: Vec<f64> = Vec::with_capacity(output);
        let mut d_output: Vec<f64> = Vec::with_capacity(output);
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
            d_output.push(desired_output);

            // if train, first output will be vector of error
            if train {
                outputs[i] = desired_output - outputs[i];
            }
        }
        (outputs, d_output)
    }
}

pub fn cross_validation(
    (input, hidden_layer, output): (usize, Vec<usize>, usize),
    (population, elitism_number, min, pm): (usize, usize, f64, f64),
    validate_section: usize,
    epoch: usize,
    data: Vec<Vec<f64>>,
    out: String,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let path = format!("./out/{}", out);
    let path = Path::new(&path);
    let display = path.display();

    let mut out: Vec<Vec<f64>> = Vec::new();
    let mut d_out: Vec<Vec<f64>> = Vec::new();

    let mut f = match File::create(&path) {
        Err(why) => panic!("couldn't create {}: {}", display, why.description()),
        Ok(f) => f,
    };

    let k: i32 = thread_rng().gen_range(1, 5);
    let data_sections = function::split_section(data, validate_section);
    let master_chromosomes = NeuralNetwork::populate(input, &hidden_layer, output, population);
    for i in 0..validate_section {
        let mut chromosomes = ga::clone_population(&master_chromosomes);
        // total of epoch * number of lines generations

        {
            if let Err(why) = f.write_all(format!("\
                ========================================================================================\n\
                BEFORE\n").as_bytes()) {
                panic!("couldn't write to {}: {}", display, why.description());
            }
            for i in 0..chromosomes.len() {
                if let Err(why) = f.write_all(format!("{}\n", chromosomes[i]).as_bytes()) {
                    panic!(
                        "couldn't write chromosome {} to {}: {}",
                        i,
                        display,
                        why.description()
                    );
                }
            }
        }

        for _iter in 0..epoch {
            for j in 0..validate_section {
                // skip index i
                if j == i {
                    continue;
                }

                let data = &data_sections[j];
                for item in data.iter() {
                    // collection of fitness/ chromosome index pair
                    let mut fitnesses: Vec<(f64, usize)> = Vec::with_capacity(population);

                    for (index, chromosome) in chromosomes.iter().enumerate() {
                        let (errors, _desired_output) =
                            chromosome.forward_pass(item, (input, &hidden_layer, output), true);
                        fitnesses.push((ga::fitness(errors, k), index));
                    }
                    fitnesses.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
                    assert_eq!(
                        fitnesses.len(),
                        population,
                        "section {} fitness {:?}",
                        j,
                        fitnesses
                    );
                    let mut next_gen = ga::elitism(&chromosomes, elitism_number, &fitnesses);
                    let mut mating_pool =
                        ga::selection(&mut chromosomes, population, min, &fitnesses);
                    let p2 = ga::recombination(mating_pool);
                    let mut p2 = ga::mutate(p2, pm);
                    loop {
                        if next_gen.len() < population {
                            let i = thread_rng().gen_range(0, p2.len());
                            let sel: NeuralNetwork = p2[i].clone();
                            p2.remove(i);
                            next_gen.push(sel)
                        } else {
                            break;
                        }
                    }
                    chromosomes = next_gen;
                }
            }
        }

        let data = &data_sections[i];

        // first chromosome is the fittest one, as being kept by elitism
        let chromosome: NeuralNetwork = chromosomes[0].clone();

        // find fittest chromosome & validity test of fittest one
        {
            for item in data.iter() {
                let (output, desired_output) =
                    chromosome.forward_pass(item, (input, &hidden_layer, output), false);
                assert_eq!(output.len(), desired_output.len());
                out.push(output);
                d_out.push(desired_output);
            }

            if let Err(why) = f.write_all(format!("\
                ========================================================================================\n\
                BEST Chromosome in fold #{}\n{}\n", i, chromosome).as_bytes()) {
                panic!("couldn't write to {}: {}", display, why.description());
            }
        }
    }

    // for classification
    (out, d_out)
}
