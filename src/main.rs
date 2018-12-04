extern crate genetic_algorithm;

use std::fs::File;
use std::io::BufReader;
use genetic_algorithm::*;

// 8 experiments; 10 and 20 populations 
// with 1 and 2 hidden layers 
// on 200 and 500 epochs
fn main() {
    let _expr1: Vec<usize> = vec![3];
    let _expr2: Vec<usize> = vec![3, 2];
    let filename = "./data/wdbc.data";
    println!("In file {}", filename);

    let f = File::open(filename).expect("File not found");
    let f = BufReader::new(f);

    let input = data_ops::vectorize(f);
    // let (_populations, _neuron_type) = neural_network::NeuralNetwork::populate(30, _expr1, 2, 10);
    neural_network::cross_validation((30, _expr2, 2), 10, 10, 200, input);
}
