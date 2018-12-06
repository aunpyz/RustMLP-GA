use super::NeuralNetwork;
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

pub fn recombination(mp: &mut Vec<NeuralNetwork>) -> Vec<NeuralNetwork> {
    let mut children: Vec<NeuralNetwork> = Vec::with_capacity(mp.len());
    loop {
        if mp.is_empty() || mp.len() == 1 {
            break;
        }

        let c1 = match mp.pop() {
            Some(data) => data,
            None => panic!("Unexpected error"),
        };
        let c2: NeuralNetwork;

        let len = mp.len();
        if len > 1 {
            let mate_index: usize = thread_rng().gen_range(0, mp.len());
            c2 = mp[mate_index].clone();
            mp.remove(mate_index);
        } else {
            c2 = mp[0].clone();
            mp.clear();
        }

        let (c1, c2) = crossover(c1, c2);
        children.extend_from_slice(&[c1, c2]);
    }
    children
}

// 2-point crossover
fn crossover(c1: NeuralNetwork, c2: NeuralNetwork) -> (NeuralNetwork, NeuralNetwork) {
    let range = c1.weight_len();
    let mut rng = thread_rng();

    let site1: usize = rng.gen_range(0, range);
    let site2: usize = rng.gen_range(site1, range);

    let w1 = c1.weights;
    let w2 = c2.weights;
    {
        let mut new_w1 = w1[0..site1].to_vec();
        new_w1.extend_from_slice(&w2[site1..site2]);
        new_w1.extend_from_slice(&w1[site2..range]);

        let mut new_w2 = w2[0..site1].to_vec();
        new_w2.extend_from_slice(&w1[site1..site2]);
        new_w2.extend_from_slice(&w2[site2..range]);

        (
            NeuralNetwork { weights: new_w1 },
            NeuralNetwork { weights: new_w2 },
        )
    }
}
