pub fn sigmoid(t: f64) -> f64 {
    1_f64 / (1_f64 + (-t).exp())
}

pub fn split_section(data: Vec<Vec<f64>>, section: usize) -> Vec<Vec<Vec<f64>>> {
    let len = data.len();
    let n = len / section;
    let mut split_data: Vec<Vec<Vec<f64>>> = Vec::new();
    for i in 0..section {
        // range slice start..end, from start to, not including, end
        if i == section - 1 {
            // last data chunk
            split_data.push(data[i * n..len].to_vec());
        } else {
            split_data.push(data[i * n..(i + 1) * n].to_vec());
        }
    }
    split_data
}

pub fn mean_sqrt_err(error_vec: Vec<f64>) -> f64 {
    let mut sse: f64 = 0 as f64;
    for (_i, item) in error_vec.iter().enumerate() {
        sse += item.powi(2);
    }
    sse / 2.0
}
