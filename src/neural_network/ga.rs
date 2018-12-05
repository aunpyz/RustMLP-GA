use super::NeuralNetwork;
use std::collections::HashMap;
use rand::{thread_rng, Rng};

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
pub fn selection(
    population: &mut Vec<NeuralNetwork>,
    n: usize,
    min: f64,
    fitness: &Vec<(f64, usize)>,
) -> Vec<NeuralNetwork> {
    let max = 2_f64 - min;
    let mut mp: Vec<NeuralNetwork> = Vec::with_capacity(n);
    let mut p: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        p.push(
            n as f64 * ((1_f64 / n as f64) * (min + (max - min) * (i as f64 / (n as f64 - 1_f64)))),
        );
    }

    // vector storing moved population index
    let mut moved: Vec<usize> = Vec::new();
    let mut ptr = thread_rng().gen_range(0_f64, 1_f64);
    let mut sum = 0_f64;
    for i in 0..n {
        let mut picked: bool = false;
        sum += p[i];
        loop {
            if sum <= ptr {
                if picked {
                    moved.push(i);
                }
                break;
            }
            picked = true;
            mp.push(population[fitness[i].1].clone());
            ptr += 1_f64;
        }
    }

    moved.reverse();
    // remove chromosomes taken to mating pool
    for i in 0..moved.len() {
        population.remove(moved[i]);
    }
    assert_eq!(mp.len(), n);
    mp
}
