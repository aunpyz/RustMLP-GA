use super::NeuralNetwork;

pub fn fitness(error_vec: Vec<f64>) -> f64 {
    let mut sse: f64 = 0 as f64;
    for (_i, item) in error_vec.iter().enumerate() {
        sse += item.powi(2);
    }
    sse / -2.0
}

pub fn clone_population(population: &Vec<NeuralNetwork>) -> Vec<NeuralNetwork> {
    let mut copied: Vec<NeuralNetwork> = Vec::new();
    for chromosome in population {
        copied.push(chromosome.clone());
    }
    copied
}
