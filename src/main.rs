extern crate genetic_algorithm;

use std::fs::File;
use std::io::BufReader;
use genetic_algorithm::{data_ops, neural_network};

// 4 experiments; 50 and 100 populations with 1 and 2 hidden layers
fn main() {
    let _expr1: Vec<usize> = vec![3];
    let _expr2: Vec<usize> = vec![3, 2];
    let filename = "./data/wdbc.data";
    println!("In file {}", filename);

    let f = File::open(filename).expect("File not found");
    let f = BufReader::new(f);

    let _input = data_ops::vectorize(f);
    let _populations = neural_network::NeuralNetwork::populate(30, _expr2, 2, 10);
}
