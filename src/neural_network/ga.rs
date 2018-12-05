use super::NeuralNetwork;

pub fn fitness(error_vec: Vec<f64>, k: i32) -> f64 {
    let mut sse: f64 = 0 as f64;
    for (_i, item) in error_vec.iter().enumerate() {
        sse += item.powi(2);
    }
    let fx = sse / 2.0;
    1.0 / (k as f64 + fx)
}

pub fn clone_population(population: &Vec<NeuralNetwork>) -> Vec<NeuralNetwork> {
    let mut copied: Vec<NeuralNetwork> = Vec::new();
    for chromosome in population {
        copied.push(chromosome.clone());
    }
    copied
}

// sorted fitness as input
pub fn elitism(
    population: &Vec<NeuralNetwork>,
    elitism_number: usize,
    fitness: &Vec<(f64, usize)>,
) -> Vec<NeuralNetwork> {
    // index of fittest chromosome
    let mut kept_chr: Vec<NeuralNetwork> = Vec::with_capacity(elitism_number);
    let fittest = fitness.len() - 1;
    for i in 0..elitism_number {
        kept_chr.push(population[fitness[fittest - i].1].clone());
    }

    assert_eq!(kept_chr.len(), elitism_number);
    kept_chr
}
