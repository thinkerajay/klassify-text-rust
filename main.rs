use std::cmp::{max, min};
use std::fmt::{Display, Formatter};
use std::io;
use std::io::prelude::*;
use std::path::Path;

use actix_web::{App, HttpResponse, HttpServer, post, Responder};
use lazy_static::lazy_static;
use tokio::fs::File;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::RwLock;

static GLOBAL_K: i32 = 1000;
lazy_static! {
    static ref SAMPLES: RwLock<Vec<Sample>> = RwLock::new(vec![]);
}

enum Klass {
    World = 1,
    Sports,
    Business,
    SciTech,
    Invalid,
}

impl From<usize> for Klass {
    fn from(value: usize) -> Self {
        match value {
            1 => Klass::World,
            2 => Klass::Sports,
            3 => Klass::Business,
            4 => Klass::SciTech,
            _ => Klass::Invalid
        }
    }
}

impl Display for Klass {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let res = match *self {
            Klass::World => "World",
            Klass::Business => "Business",
            Klass::SciTech => "Sci/Tech",
            Klass::Invalid => "Invalid",
            Klass::Sports => "Sports"
        };
        write!(f, "{}", res)
    }
}

struct Sample {
    klass: u8,
    text: Vec<u8>,
    original: String,
}

impl Sample {
    fn new(klass: u8, text: Vec<u8>, original: String) -> Self {
        Sample {
            klass,
            text,
            original,
        }
    }
}

async fn load_train_data() -> io::Result<()> {
    let file_ptr = File::open(Path::new("data/train.csv")).await?;
    let mut buf_reader = BufReader::new(file_ptr);

    let mut line = String::new();
    buf_reader.read_line(&mut line).await.expect("skip first line");
    line.clear();
    let mut klass_freq = [0; 5];
    println!("{line}");
    {
        let mut samples = SAMPLES.write().await;
        while let Ok(n) = buf_reader.read_line(&mut line).await {
            if n == 0 {
                println!("Reached EOF !");
                break;
            }
            let mut compressor = async_compression::tokio::write::ZlibEncoder::new(vec![]);
            let vals: Vec<&str> = line.splitn(2, ",").collect();
            let klass = vals[0].to_string();
            let text = vals[1].to_string();
            if let Ok(v) = klass.parse::<u8>() {
                if let Ok(_) = compressor.write_all(text.as_bytes()).await {
                    compressor.flush().await.expect("to shutdown");
                    klass_freq[v as usize] += 1;
                    (*samples).push(Sample::new(v, compressor.into_inner(), text));
                } else {
                    eprintln!("Compression failed");
                    break;
                }
            } else {
                eprintln!("Invalid data {}", klass);
                break;
            }
            line.clear();
        }
    }
    println!("Read {} lines with klasses {:?}", SAMPLES.read().await.len(), klass_freq);
    Ok(())
}


async fn compress_bytes(x1: &str, x2: &str) -> usize {
    let mut data = String::from(x1);
    data.push_str(x2);
    let mut compressor = async_compression::tokio::write::ZlibEncoder::new(vec![]);
    if let Ok(_) = compressor.write_all(data.as_bytes()).await {
        compressor.shutdown().await.expect("to shutdown compressor");
        compressor.into_inner().len()
    } else {
        1
    }
}

async fn klassify_text(body: &str) -> io::Result<String> {
    let samples = SAMPLES.read().await;
    let mut compressor = async_compression::tokio::write::ZlibEncoder::new(vec![]);
    let mut min_heap = Vec::new();
    if let Ok(_) = compressor.write_all(body.as_bytes()).await {
        compressor.shutdown().await.expect("to shutdown compressor");
        let cx1 = compressor.into_inner().len();
        for sample in samples.iter() {
            let cx2 = sample.text.len();
            let cx1x2 = compress_bytes(body, sample.original.as_str()).await;
            let ncd: f64 = ((cx1x2 - min(cx1, cx2)) / max(cx1, cx2)) as f64;
            min_heap.push((ncd, sample.klass));
        }
    }
    // mimic min_heap by sorting in ASC
    min_heap.sort_by(|left, right| left.0.partial_cmp(&right.0).unwrap());
    let mut klass_freq = compute_klass_freq(&mut min_heap);
    let idx = get_max_freq_klass(&mut klass_freq);
    Ok(Klass::from(idx).to_string())
}

fn compute_klass_freq(min_heap: &mut Vec<(f64, u8)>) -> [i32; 5] {
    let mut klass_freq = [0; 5];

    for (idx, val) in min_heap.iter().enumerate() {
        if idx >= GLOBAL_K as usize {
            break;
        }
        klass_freq[val.1 as usize] += 1;
    }
    klass_freq
}

fn get_max_freq_klass(klass_freq: &mut [i32; 5]) -> usize {
    // not an idiomatic way but serves the purpose
    let mut mx = 0;
    let mut idx = 0;
    for i in (0..5) {
        if klass_freq[i] > mx {
            mx = klass_freq[i];
            idx = i;
        }
    }
    idx
}


#[post("/classify")]
async fn classify_text(req_body: String) -> impl Responder {
    println!("Received {} text for classification", req_body.as_str());
    if let Ok(resp) = klassify_text(req_body.as_str()).await {
        HttpResponse::Ok().body(resp)
    } else {
        HttpResponse::BadRequest().body(req_body.to_uppercase())
    }
}

#[actix_web::main]
async fn main() -> io::Result<()> {
    load_train_data().await.expect("To load training data");
    HttpServer::new(|| {
        App::new()
            .service(classify_text)
    })
        .bind(("127.0.0.1", 4895))?
        .run()
        .await
}
