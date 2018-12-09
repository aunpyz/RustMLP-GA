extern crate genetic_algorithm;

use std::fs::File;
use std::io::BufReader;
use genetic_algorithm::*;

// 12 experiments; 10 and 20 populations
// with 1 and 2 hidden layers ([3], [5], [3, 2])
// on 200 and 500 epochs
fn main() {
	// experiment #1
    {
    	let _expr1: Vec<usize> = vec![3];
    	let filename = "./data/wdbc.data";
	    println!("In file {}", filename);

	    let f = File::open(filename).expect("File not found");
	    let f = BufReader::new(f);

	    let input = data_ops::vectorize(f);
	    let (out, desire_output) = neural_network::cross_validation(
	        (30, _expr1, 2),
	        (10, 3, 0.2, 0.01),
	        10,
	        200,
	        input,
	        String::from("wdbc_out_1_1.txt"),
	    );
	    println!("Write confusion matrix file wdbc_cross_out_1_1.txt");
	    data_ops::confusion_matrix((out, desire_output), String::from("wdbc_cross_out_1_1.txt"));
    }
    {
    	let _expr1: Vec<usize> = vec![3];
    	let filename = "./data/wdbc.data";
	    println!("In file {}", filename);

	    let f = File::open(filename).expect("File not found");
	    let f = BufReader::new(f);

	    let input = data_ops::vectorize(f);
	    let (out, desire_output) = neural_network::cross_validation(
	        (30, _expr1, 2),
	        (10, 3, 0.2, 0.01),
	        10,
	        500,
	        input,
	        String::from("wdbc_out_1_2.txt"),
	    );
	    println!("Write confusion matrix file wdbc_cross_out_1_2.txt");
	    data_ops::confusion_matrix((out, desire_output), String::from("wdbc_cross_out_1_2.txt"));
    }
        {
    	let _expr1: Vec<usize> = vec![3];
    	let filename = "./data/wdbc.data";
	    println!("In file {}", filename);

	    let f = File::open(filename).expect("File not found");
	    let f = BufReader::new(f);

	    let input = data_ops::vectorize(f);
	    let (out, desire_output) = neural_network::cross_validation(
	        (30, _expr1, 2),
	        (10, 3, 0.2, 0.01),
	        20,
	        200,
	        input,
	        String::from("wdbc_out_1_3.txt"),
	    );
	    println!("Write confusion matrix file wdbc_cross_out_1_3.txt");
	    data_ops::confusion_matrix((out, desire_output), String::from("wdbc_cross_out_1_3.txt"));
    }
    {
    	let _expr1: Vec<usize> = vec![3];
    	let filename = "./data/wdbc.data";
	    println!("In file {}", filename);

	    let f = File::open(filename).expect("File not found");
	    let f = BufReader::new(f);

	    let input = data_ops::vectorize(f);
	    let (out, desire_output) = neural_network::cross_validation(
	        (30, _expr1, 2),
	        (10, 3, 0.2, 0.01),
	        20,
	        500,
	        input,
	        String::from("wdbc_out_1_4.txt"),
	    );
	    println!("Write confusion matrix file wdbc_cross_out_1_4.txt");
	    data_ops::confusion_matrix((out, desire_output), String::from("wdbc_cross_out_1_4.txt"));
    }

    // experiment #2
    {
    	let _expr1: Vec<usize> = vec![5];
    	let filename = "./data/wdbc.data";
	    println!("In file {}", filename);

	    let f = File::open(filename).expect("File not found");
	    let f = BufReader::new(f);

	    let input = data_ops::vectorize(f);
	    let (out, desire_output) = neural_network::cross_validation(
	        (30, _expr1, 2),
	        (10, 3, 0.2, 0.01),
	        10,
	        200,
	        input,
	        String::from("wdbc_out_2_1.txt"),
	    );
	    println!("Write confusion matrix file wdbc_cross_out_2_1.txt");
	    data_ops::confusion_matrix((out, desire_output), String::from("wdbc_cross_out_2_1.txt"));
    }
    {
    	let _expr1: Vec<usize> = vec![5];
    	let filename = "./data/wdbc.data";
	    println!("In file {}", filename);

	    let f = File::open(filename).expect("File not found");
	    let f = BufReader::new(f);

	    let input = data_ops::vectorize(f);
	    let (out, desire_output) = neural_network::cross_validation(
	        (30, _expr1, 2),
	        (10, 3, 0.2, 0.01),
	        10,
	        500,
	        input,
	        String::from("wdbc_out_2_2.txt"),
	    );
	    println!("Write confusion matrix file wdbc_cross_out_2_2.txt");
	    data_ops::confusion_matrix((out, desire_output), String::from("wdbc_cross_out_2_2.txt"));
    }
        {
    	let _expr1: Vec<usize> = vec![5];
    	let filename = "./data/wdbc.data";
	    println!("In file {}", filename);

	    let f = File::open(filename).expect("File not found");
	    let f = BufReader::new(f);

	    let input = data_ops::vectorize(f);
	    let (out, desire_output) = neural_network::cross_validation(
	        (30, _expr1, 2),
	        (10, 3, 0.2, 0.01),
	        20,
	        200,
	        input,
	        String::from("wdbc_out_2_3.txt"),
	    );
	    println!("Write confusion matrix file wdbc_cross_out_2_3.txt");
	    data_ops::confusion_matrix((out, desire_output), String::from("wdbc_cross_out_2_3.txt"));
    }
    {
    	let _expr1: Vec<usize> = vec![5];
    	let filename = "./data/wdbc.data";
	    println!("In file {}", filename);

	    let f = File::open(filename).expect("File not found");
	    let f = BufReader::new(f);

	    let input = data_ops::vectorize(f);
	    let (out, desire_output) = neural_network::cross_validation(
	        (30, _expr1, 2),
	        (10, 3, 0.2, 0.01),
	        20,
	        500,
	        input,
	        String::from("wdbc_out_2_4.txt"),
	    );
	    println!("Write confusion matrix file wdbc_cross_out_2_4.txt");
	    data_ops::confusion_matrix((out, desire_output), String::from("wdbc_cross_out_2_4.txt"));
    }

    // experiment #3
    {
    	let _expr1: Vec<usize> = vec![3, 2];
    	let filename = "./data/wdbc.data";
	    println!("In file {}", filename);

	    let f = File::open(filename).expect("File not found");
	    let f = BufReader::new(f);

	    let input = data_ops::vectorize(f);
	    let (out, desire_output) = neural_network::cross_validation(
	        (30, _expr1, 2),
	        (10, 3, 0.2, 0.01),
	        10,
	        200,
	        input,
	        String::from("wdbc_out_3_1.txt"),
	    );
	    println!("Write confusion matrix file wdbc_cross_out_3_1.txt");
	    data_ops::confusion_matrix((out, desire_output), String::from("wdbc_cross_out_3_1.txt"));
    }
    {
    	let _expr1: Vec<usize> = vec![3, 2];
    	let filename = "./data/wdbc.data";
	    println!("In file {}", filename);

	    let f = File::open(filename).expect("File not found");
	    let f = BufReader::new(f);

	    let input = data_ops::vectorize(f);
	    let (out, desire_output) = neural_network::cross_validation(
	        (30, _expr1, 2),
	        (10, 3, 0.2, 0.01),
	        10,
	        500,
	        input,
	        String::from("wdbc_out_3_2.txt"),
	    );
	    println!("Write confusion matrix file wdbc_cross_out_3_2.txt");
	    data_ops::confusion_matrix((out, desire_output), String::from("wdbc_cross_out_3_2.txt"));
    }
        {
    	let _expr1: Vec<usize> = vec![3, 2];
    	let filename = "./data/wdbc.data";
	    println!("In file {}", filename);

	    let f = File::open(filename).expect("File not found");
	    let f = BufReader::new(f);

	    let input = data_ops::vectorize(f);
	    let (out, desire_output) = neural_network::cross_validation(
	        (30, _expr1, 2),
	        (10, 3, 0.2, 0.01),
	        20,
	        200,
	        input,
	        String::from("wdbc_out_3_3.txt"),
	    );
	    println!("Write confusion matrix file wdbc_cross_out_3_3.txt");
	    data_ops::confusion_matrix((out, desire_output), String::from("wdbc_cross_out_3_3.txt"));
    }
    {
    	let _expr1: Vec<usize> = vec![3, 2];
    	let filename = "./data/wdbc.data";
	    println!("In file {}", filename);

	    let f = File::open(filename).expect("File not found");
	    let f = BufReader::new(f);

	    let input = data_ops::vectorize(f);
	    let (out, desire_output) = neural_network::cross_validation(
	        (30, _expr1, 2),
	        (10, 3, 0.2, 0.01),
	        20,
	        500,
	        input,
	        String::from("wdbc_out_3_4.txt"),
	    );
	    println!("Write confusion matrix file wdbc_cross_out_3_4.txt");
	    data_ops::confusion_matrix((out, desire_output), String::from("wdbc_cross_out_3_4.txt"));
    }
}
