use std::{fs, io};
use chrono::NaiveDateTime;
use clap::Parser;
use rustfft::{FftPlanner, num_complex::Complex};
use serde::Deserialize;

type DataPoint = f32;

#[derive(Parser)]
struct Cli {
    /// Path to the input CSV file. If not provided or "-", it reads from the standard input.
    path: Option<String>,

    /// Window size for calculating a saliency map.
    #[clap(short, long, default_value = "3")]
    q: usize,

    /// Window size for calculating the average of the saliency map which is used for scoring.
    #[clap(short, long, default_value = "21")]
    z: usize,

    /// Threshold that determines if a data point is an anomaly.
    #[clap(short, long, default_value = "3.0")]
    t: DataPoint,

    /// Number of preceding points considered for extrapolation.
    #[clap(short, long, default_value = "5")]
    m: usize,

    /// Number of extrapolated points. 0 for disabling extrapolation.
    #[clap(short, long, default_value = "5")]
    k: usize,
}

#[derive(Debug, Deserialize, PartialEq)]
struct Record {
    #[serde(rename = "Time", with = "timestamp_format")]
    time: NaiveDateTime,
    value: DataPoint,
}

fn main() {
    let cli = Cli::parse();

    let input: Box<dyn io::BufRead> = match cli.path.as_deref() {
        Some("-") | None => {
            Box::new(io::BufReader::new(io::stdin()))
        }
        Some(path) => {
            fs::File::open(path).map(io::BufReader::new).map(Box::new).unwrap()
        },
    };

    // hyperparameters
    let q = cli.q;
    let z = cli.z;
    let t = cli.t;
    let m = cli.m;
    let k = cli.k;

    let records = read_data(input);
    let (times, data): (Vec<_>, Vec<_>) = records.iter().map(|r| (r.time, r.value)).unzip();
    let (map, score, anomalies) = detect(&data, q, z, t, m, k);
    println!("Time,value,saliency,score,output");
    for ((((time, value), spectrum), score), anomaly) in times.iter().zip(data.iter()).zip(map.iter()).zip(score.iter()).zip(anomalies.iter()) {
        println!("{},{},{},{},{}", time, value, spectrum, score, if *anomaly { 1 } else { 0 });
    }
    // println!("found {} anomalies in {} records", anomalies.iter().filter(|x| **x).count(), records.len());
}

fn read_data(input: Box<dyn io::BufRead>) -> Vec<Record> {
    let mut rdr = csv::Reader::from_reader(input);
    rdr.deserialize().map(|result| result.unwrap()).collect()
}

/// Detects the anomalies in the `data` using Spectral Residual method.
/// `q` is the window size for calculating a saliency map.
/// `z` is the window size for calculating the average of the saliency map which is used for scoring.
/// `t` is the threshold that determines if a data point is an anomaly.
/// `m` is the number of preceding points considered for extrapolation.
/// `k` is the number of extrapolated points.
/// Returns a vector of booleans where `true` indicates an anomaly. Its size is the same as the input `data`.
fn detect(data: &[DataPoint], q: usize, z: usize, t: DataPoint, m: usize, k: usize) -> (Vec<DataPoint>, Vec<DataPoint>, Vec<bool>) {
    let n = data.len();
    let data = extrapolate(data, m, k);
    // cut the extrapolated points
    let saliency_map = calculate_saliency_map(&data, q)[k..(n+k)].to_vec();
    let score = calculate_score(&saliency_map, z);
    let result = score.iter().map(|&x| x > t).collect();
    (saliency_map, score, result)
}

/// Extrapolates the data.
/// The extrapolated point x_(n+1) is calculated by x_(n-m+1) + g * m,
/// where g is the average gradient of the last m points, and m is the number of preceding points considered.
/// `k` points are extrapolated.
/// If `k` is 0, it returns the original data.
fn extrapolate(data: &[DataPoint], m: usize, k: usize) -> Vec<DataPoint> {
    if k == 0 {
        return data.to_vec();
    }
    assert!(m <= data.len(), "m must be less than or equal to the length of the data");
    let last_idx = data.len() - 1;
    let g = (last_idx.wrapping_sub(m)..last_idx).map(|i| gradient(data, last_idx, i)).sum::<DataPoint>() / m as DataPoint;
    let extra_value = data[last_idx.wrapping_sub(m).wrapping_add(1)] + g * m as DataPoint;
    [vec![extra_value; k], data.to_vec(), vec![extra_value; k]].concat()
}

/// Calculates the gradient of the two points.
fn gradient(data: &[DataPoint], x1: usize, x2: usize) -> DataPoint {
    assert!(x1 != x2);
    (data[x2] - data[x1]) / x2.wrapping_sub(x1) as DataPoint
}

/// Calculates a saliency map.
fn calculate_saliency_map(data: &[DataPoint], q: usize) -> Vec<DataPoint> {
    // perform FFT
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(data.len());
    let mut freq = data.iter().map(|x| Complex::new(*x, 0.0)).collect::<Vec<_>>();
    fft.process(&mut freq);

    // calculate spectral residual
    let (amp, phase): (Vec<DataPoint>, Vec<DataPoint>) = freq.iter().map(|x| x.scale(1.0 / (data.len() as DataPoint)).to_polar()).unzip();
    let log_amp = amp.iter().map(|x| x.ln()).collect::<Vec<_>>();
    let average_log_amp = convolve(&log_amp, q);
    let spectral_residual = log_amp.iter().zip(average_log_amp.iter()).map(|(&x, &y)| x - y).collect::<Vec<_>>();

    // perform IFFT
    let ifft = planner.plan_fft_inverse(data.len());
    let mut saliency_map = spectral_residual.iter().zip(phase.iter()).map(|(&r, &p)| Complex::from_polar(r, p).exp()).collect::<Vec<_>>();
    ifft.process(&mut saliency_map);

    // saliency map is the norm of the IFFT result
    saliency_map.iter().map(|x| x.norm()).collect()
}

/// Calculates the scores of the saliency map.
/// The scores are calculated by: (S - S_average) / S_average; where S is the saliency map and S_average is the local-averaged saliency map using window `z`.
fn calculate_score(saliency_map: &[DataPoint], z: usize) -> Vec<DataPoint> {
    let averaged_saliency_map = convolve(saliency_map, z);
    saliency_map.iter().zip(averaged_saliency_map.iter()).map(|(&s, &sa)| (s - sa) / sa).collect()
}

/// Convolve using a window size of `w`.
fn convolve(data: &[DataPoint], w: usize) -> Vec<DataPoint> {
    fn is_even(n: usize) -> bool {
        n % 2 == 0
    }
    // add padding to the both ends of the data
    let lp = if is_even(w) { w / 2 - 1 } else { (w - 1) / 2 };
    let rp = w - lp - 1;
    [vec![0.0; lp], data.to_vec(), vec![0.0; rp]].concat()
        .windows(w)
        .map(|xs| xs.iter().sum::<DataPoint>() / w as DataPoint)
        .collect()
}

pub mod timestamp_format {
    use chrono::{DateTime, NaiveDateTime};
    use serde::{self, Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(datetime: &NaiveDateTime, serializer: S) -> Result<S::Ok, S::Error> {
        let s = format!("{}", datetime.and_utc().timestamp_millis());
        serializer.serialize_str(&s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<NaiveDateTime, D::Error> {
        let ts = i64::deserialize(deserializer)?;
        DateTime::from_timestamp_millis(ts).map(|dt| dt.naive_utc()).ok_or(serde::de::Error::custom("invalid timestamp"))
    }
}

pub mod datetime_format {
    use chrono::NaiveDateTime;
    use serde::{self, Deserialize, Deserializer, Serializer};

    const FORMAT: &str = "%Y-%m-%d %H:%M:%S";

    pub fn serialize<S: Serializer>(datetime: &NaiveDateTime, serializer: S) -> Result<S::Ok, S::Error> {
        let s = format!("{}", datetime.format(FORMAT));
        serializer.serialize_str(&s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<NaiveDateTime, D::Error> {
        let s = String::deserialize(deserializer)?;
        NaiveDateTime::parse_from_str(&s, FORMAT).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use chrono::NaiveDate;

    #[test]
    fn test_read_data() {
        let data = "\"Time\",\"value\"\n1732163400000,67553\n1732163520000,18875\n1732163640000,0".to_string();
        let cursor = io::Cursor::new(data);
        let input = Box::new(io::BufReader::new(cursor));
        let result = read_data(input);
        assert_eq!(result.len(), 3);
        assert_eq!(result, vec![
            Record { time: NaiveDate::from_ymd_opt(2024, 11, 21).unwrap().and_hms_opt(4, 30, 0).unwrap() , value: 67553.0 },
            Record { time: NaiveDate::from_ymd_opt(2024, 11, 21).unwrap().and_hms_opt(4, 32, 0).unwrap() , value: 18875.0 },
            Record { time: NaiveDate::from_ymd_opt(2024, 11, 21).unwrap().and_hms_opt(4, 34, 0).unwrap() , value: 0.0 },
        ]);
    }

    #[test]
    fn test_detect() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (map, score, result) = detect(&data, 3, 5, 2.0, 5, 3);
        assert_eq!(map.len(), data.len());
        assert_eq!(score.len(), data.len());
        assert_eq!(result.len(), data.len());
    }

    #[test]
    fn test_extrapolate() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = extrapolate(&data, 3, 2);
        let expected = [3.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 3.0, 3.0];
        assert_eq!(result.len(), expected.len());
        for i in 0..result.len() {
            assert_relative_eq!(result[i], expected[i]);
        }
    }

    #[test]
    fn test_calculate_saliency_map() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = calculate_saliency_map(&data, 3);
        assert_eq!(result.len(), data.len());
    }

    #[test]
    fn test_calculate_score() {
        let saliency_map = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = calculate_score(&saliency_map, 3);
        assert_eq!(result.len(), saliency_map.len());
    }

    #[test]
    fn test_convolve() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = convolve(&data, 3);
        let expected = [1.0, 2.0, 3.0, 4.0, 3.0];
        assert_eq!(result.len(), data.len());
        for i in 0..result.len() {
            assert_relative_eq!(result[i], expected[i]);
        }
    }

    #[test]
    fn  test_convolve_with_large_window() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = convolve(&data, 5);
        let expected = [1.2, 2.0, 3.0, 2.8, 2.4];
        assert_eq!(result.len(), data.len());
        for i in 0..result.len() {
            assert_relative_eq!(result[i], expected[i]);
        }
    }

    #[test]
    fn test_convolve_with_padding_even() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = convolve(&data, 4);
        let expected = [1.5, 2.5, 3.5, 3.0, 2.25];
        assert_eq!(result.len(), data.len());
        for i in 0..result.len() {
            assert_relative_eq!(result[i], expected[i]);
        }
    }

}
