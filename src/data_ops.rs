use rand::seq::SliceRandom;
use std::fs::File;
use std::path::Path;
use std::error::Error;
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

// example of data in output, desire_output vector
// [[0,1], [1,0], [0,1], ...]

// output structure
// output/ desire output   M   B    undefined
//                     M   #   #       #
//                     B   #   #       #
//             undefined   #   #       #
pub fn confusion_matrix(
    (output, desire_output): (Vec<Vec<f64>>, Vec<Vec<f64>>),
    out_filename: String,
) {
    assert_eq!(output.len(), desire_output.len());
    assert_eq!(output[0].len(), 2);

    let path = format!("./out/{}", out_filename);
    let path = Path::new(&path);
    let display = path.display();

    let mut matrix: Vec<Vec<usize>> = vec![vec![0, 0, 0], vec![0, 0, 0], vec![0, 0, 0]];
    let mut f = match File::create(&path) {
        Err(why) => panic!("couldn't create {}: {}", display, why.description()),
        Ok(f) => f,
    };

    for i in 0..output.len() {
        let row = if output[i][0] < output[i][1] {
            // M
            0
        } else if output[i][0] > output[i][1] {
            // B
            1
        } else {
            // undefined class
            2
        };
        // desire output classes are checked in preprocess; vectorize function
        let col = if desire_output[i][0] < desire_output[i][1] {
            0
        } else {
            1
        };

        matrix[row][col] += 1;
    }

    {
        if let Err(why) = f.write_all(
            format!(
                "\
            output\\desire output\t|\tM\t|\tB\t|\tundefined\n\
            M\t\t\t\t\t\t|\t{}\t|\t{}\t|\t{}\n\
            B\t\t\t\t\t\t|\t{}\t|\t{}\t|\t{}\n\
            undefined\t\t\t\t|\t{}\t|\t{}\t|\t{}\n\
            =====================================================\n",
                matrix[0][0],
                matrix[0][1],
                matrix[0][2],
                matrix[1][0],
                matrix[1][1],
                matrix[1][2],
                matrix[2][0],
                matrix[2][1],
                matrix[2][2]
            ).as_bytes(),
        )
        {
            panic!("couldn't write to {}: {}", display, why.description());
        }
    }
}
