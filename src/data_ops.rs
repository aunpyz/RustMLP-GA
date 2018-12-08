use rand::seq::SliceRandom;
use std::fs::File;
use std::io::BufReader;
use std::io::prelude::*;

// M = malignant => (0, 1)
// B = benign => (1, 0)
pub fn vectorize(f: BufReader<File>) -> Vec<Vec<f64>> {
    let mut data_all: Vec<Vec<f64>> = Vec::new();
    for line in f.lines() {
        let mut data_line: Vec<f64> = Vec::new();
        let line = line.unwrap();
        let data = line.split(",").collect::<Vec<&str>>();
        for i in 0..data.len() {
            if i == 0 {
                continue;
            } else if i == 1 {
                match data[i] {
                    "M" => data_line.extend_from_slice(&[0_f64, 1_f64]),
                    "B" => data_line.extend_from_slice(&[1_f64, 0_f64]),
                    _ => panic!("Mismatched type"),
                }
            } else {
                data_line.push(data[i].parse::<f64>().unwrap());
            }
        }
        data_all.push(data_line);
    }
    data_all.shuffle(&mut rand::thread_rng());
    data_all
}

pub fn confusion_matrix(
    (output, desire_output): (Vec<Vec<f64>>, Vec<Vec<f64>>),
    out_filename: String,
) {
    println!("{} {}", output.len(), desire_output.len());
    panic!("output {:?}\ndesired output {:?}", output, desire_output);
}
