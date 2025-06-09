use std::f64::consts::PI;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::ops;
use std::{path::Path};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone)]
struct Vecf {
	v: Vec<f64>,
	n: usize
}

impl Vecf {
	fn new(v: Vec<f64>) -> Vecf {
		let len = v.len();
		return Vecf {v, n: len};
	}

	fn len(&self) -> f64 {
		let sum: f64 = self.v.iter().map(|a| a * a).sum::<f64>().sqrt();
		return sum;
	}

	fn as_veci(&self) -> Veci {
		let nv: Vec<i64> = self.v.iter().map(|a| *a as i64).collect();
		let len = nv.len();
		return Veci {v: nv, n: len};
	}
}

impl ops::Index<usize> for Vecf {
	type Output = f64;

	fn index(&self, index: usize) -> &Self::Output {
		return if index < self.n {&self.v[index]} else {&0.0};
	}
}

impl ops::IndexMut<usize> for Vecf {
	fn index_mut(&mut self, index: usize) -> &mut Self::Output {
		return &mut self.v[index];
	}
}

impl ops::Add for Vecf {
	type Output = Vecf;

	fn add(self, o: Vecf) -> Vecf {
		let nv: Vec<f64> = self.v.iter().zip(o.v.iter()).map(|(a, b)| a + b).collect();
		let len = nv.len();
		return Vecf {v: nv, n: len};
	}
}

impl ops::Add for &Vecf {
	type Output = Vecf;

	fn add(self, o: &Vecf) -> Vecf {
		let nv: Vec<f64> = self.v.iter().zip(o.v.iter()).map(|(a, b)| a + b).collect();
		let len = nv.len();
		return Vecf {v: nv, n: len};
	}
}

impl ops::Add<f64> for Vecf {
	type Output = Vecf;

	fn add(self, o: f64) -> Vecf {
		let nv: Vec<f64> = self.v.iter().map(|a| a + o).collect();
		let len = nv.len();
		return Vecf {v: nv, n: len};
	}
}

impl ops::Add<f64> for &Vecf {
	type Output = Vecf;

	fn add(self, o: f64) -> Vecf {
		let nv: Vec<f64> = self.v.iter().map(|a| a + o).collect();
		let len = nv.len();
		return Vecf {v: nv, n: len};
	}
}

impl ops::Sub for Vecf {
	type Output = Vecf;

	fn sub(self, o: Vecf) -> Vecf {
		let nv: Vec<f64> = self.v.iter().zip(o.v.iter()).map(|(a, b)| a - b).collect();
		let len = nv.len();
		return Vecf {v: nv, n: len};
	}
}

impl ops::Sub for &Vecf {
	type Output = Vecf;

	fn sub(self, o: &Vecf) -> Vecf {
		let nv: Vec<f64> = self.v.iter().zip(o.v.iter()).map(|(a, b)| a - b).collect();
		let len = nv.len();
		return Vecf {v: nv, n: len};
	}
}

impl ops::Sub<f64> for Vecf {
	type Output = Vecf;

	fn sub(self, o: f64) -> Vecf {
		let nv: Vec<f64> = self.v.iter().map(|a| a - o).collect();
		let len = nv.len();
		return Vecf {v: nv, n: len};
	}
}

impl ops::Sub<f64> for &Vecf {
	type Output = Vecf;

	fn sub(self, o: f64) -> Vecf {
		let nv: Vec<f64> = self.v.iter().map(|a| a - o).collect();
		let len = nv.len();
		return Vecf {v: nv, n: len};
	}
}

impl ops::Mul<f64> for Vecf {
	type Output = Vecf;

	fn mul(self, o: f64) -> Vecf {
		let nv: Vec<f64> = self.v.iter().map(|a| a * o).collect();
		let len = nv.len();
		return Vecf {v: nv, n: len};
	}
}

impl ops::Mul<f64> for &Vecf {
	type Output = Vecf;

	fn mul(self, o: f64) -> Vecf {
		let nv: Vec<f64> = self.v.iter().map(|a| a * o).collect();
		let len = nv.len();
		return Vecf {v: nv, n: len};
	}
}

impl ops::Div<f64> for Vecf {
	type Output = Vecf;

	fn div(self, o: f64) -> Vecf {
		let nv: Vec<f64> = self.v.iter().map(|a| a / o).collect();
		let len = nv.len();
		return Vecf {v: nv, n: len};
	}
}

impl ops::Div<f64> for &Vecf {
	type Output = Vecf;

	fn div(self, o: f64) -> Vecf {
		let nv: Vec<f64> = self.v.iter().map(|a| a / o).collect();
		let len = nv.len();
		return Vecf {v: nv, n: len};
	}
}

//dot product
impl ops::BitXor for Vecf {
	type Output = f64;

	fn bitxor(self, o: Self) -> f64 {
		let sum: f64 = self.v.iter().zip(o.v.iter()).map(|(a, b)| a * b).sum();
		return sum;
	}
}

impl ops::BitXor for &Vecf {
	type Output = f64;

	fn bitxor(self, o: Self) -> f64 {
		let sum: f64 = self.v.iter().zip(o.v.iter()).map(|(a, b)| a * b).sum();
		return sum;
	}
}

impl ops::BitXor<&Vecf> for Vecf {
	type Output = f64;

	fn bitxor(self, o: &Vecf) -> f64 {
		let sum: f64 = self.v.iter().zip(o.v.iter()).map(|(a, b)| a * b).sum();
		return sum;
	}
}

impl ops::BitXor<Vecf> for &Vecf {
	type Output = f64;

	fn bitxor(self, o: Vecf) -> f64 {
		let sum: f64 = self.v.iter().zip(o.v.iter()).map(|(a, b)| a * b).sum();
		return sum;
	}
}

impl ops::Neg for Vecf {
	type Output = Vecf;

	fn neg(self) -> Vecf {
		let nv: Vec<f64> = self.v.iter().map(|a| -a).collect();
		let len = nv.len();
		return Vecf {v: nv, n: len};
	}
}

//normalize
impl ops::Not for Vecf {
	type Output = Vecf;

	fn not(self) -> Vecf {
		let veclen = self.len();
		let nv: Vec<f64> = self.v.iter().map(|a| a / veclen).collect();
		let len = nv.len();
		return Vecf {v: nv, n: len};
	}
}

struct Veci {
	v: Vec<i64>,
	n: usize
}

impl Veci {
	fn new(v: Vec<i64>) -> Veci {
		let len = v.len();
		return Veci {v, n: len};
	}

	fn as_vecf(&self) -> Vecf {
		let nv: Vec<f64> = self.v.iter().map(|a| *a as f64).collect();
		let len = nv.len();
		return Vecf {v: nv, n: len};
	}
}

impl ops::Div<i64> for Veci {
	type Output = Veci;

	fn div(self, o: i64) -> Veci {
		let nv: Vec<i64> = self.v.iter().map(|a| a / o).collect();
		let len = nv.len();
		return Veci {v: nv, n: len};
	}
}

impl ops::Div<i64> for &Veci {
	type Output = Veci;

	fn div(self, o: i64) -> Veci {
		let nv: Vec<i64> = self.v.iter().map(|a| a / o).collect();
		let len = nv.len();
		return Veci {v: nv, n: len};
	}
}

fn smoothstep(x: f64) -> f64 {
	return 3.0 * x * x - 2.0 * x * x * x;
}

fn d_smoothstep(x: f64) -> f64 {
	return 6.0 * x - 6.0 * x * x;
}

fn int_hash(i: i64) -> i64 {
	let mut x = ((i >> 32) ^ i) % -4776276827692571787;
    x = ((x >> 32) ^ x) % -4776276827692571787;
    x = (x >> 32)  ^ x;
	return x;
}

fn int_vec_hash(vec: &Veci, start: i64) -> i64 {
	let mut hash = start;
	for v in &vec.v {
		hash = int_hash(hash % -4776276827692571787 + *v as i64);
	}
	return hash;
}

fn get_gridvalue_n(pos: &Veci) -> Vecf {
	let mut bm: Vec<f64> = vec![0.0; pos.n];

	for (i, _) in pos.v.iter().enumerate() {
		// let t = (int_vec_hash(&pos, i as i64 * pos.n as i64) % 256) as f64 / 256.0 * 2.0 * PI;
		// let r = (((int_vec_hash(&pos, i as i64 * pos.n as i64 + 1) % 256 + 1) as f64 / 256.0).ln() * -2.0).sqrt();
		// bm[i] = r * t.cos();
		bm[i] = (int_vec_hash(&pos, i as i64 * pos.n as i64) % 256) as f64 / 255.0 * 2.0 - 1.0;
	}

	let out = Vecf::new(bm);

	
	return out;
}

// fn get_perlin(x: f64, y: f64) -> f64 {
//     let ffv = vec2(get_gridvalue(x.floor() as i32, y.floor() as i32) * PI * 2.0);
//     let fcv = vec2(get_gridvalue(x.floor() as i32, y.floor() as i32 + 1) * PI * 2.0);
//     let cfv = vec2(get_gridvalue(x.floor() as i32 + 1, y.floor() as i32) * PI * 2.0);
//     let ccv = vec2(get_gridvalue(x.floor() as i32 + 1, y.floor() as i32 + 1) * PI * 2.0);

//     let h1 = x - x.floor();
//     let v1 = y - y.floor();

//     let ff = Vec2 {x: h1, y: v1} ^ ffv;
//     let fc = Vec2 {x: h1, y: v1 - 1.0} ^ fcv;
//     let cf = Vec2 {x: h1 - 1.0, y: v1} ^ cfv;
//     let cc = Vec2 {x: h1 - 1.0, y: v1 - 1.0} ^ ccv;

//     let h = smoothstep(h1);
//     let v = smoothstep(v1);

//     let top = ff + (cf - ff) * h;
//     let bot = fc + (cc - fc) * h;
//     let total = top + (bot - top) * v;

//     return total * 0.5 + 0.5;
// }

fn get_perlin_n_inner(pos: &Vecf, origin: &Vecf, f_dims: usize) -> f64 {
	if f_dims == 0 {
		let v = get_gridvalue_n(&pos.as_veci());
		let cv = origin - pos;
		let val = cv ^ v;
		return val;
	} else {
		let mut fpos = pos.clone();
		let mut cpos = pos.clone();
		fpos[f_dims - 1] = pos[f_dims - 1].floor();
		cpos[f_dims - 1] = pos[f_dims - 1].ceil();

		let f = get_perlin_n_inner(&fpos, origin, f_dims - 1);
		let c = get_perlin_n_inner(&cpos, origin, f_dims - 1);

		let interp = smoothstep(pos[f_dims - 1] - pos[f_dims - 1].floor());
		let total = f + (c - f) * interp;

		return total;
	}
}

fn get_perlin_n(pos: &Vecf) -> f64 {
	let val = get_perlin_n_inner(pos, pos, pos.n);

	return val;
}

fn get_perlin_n_grad_inner(pos: &Vecf, origin: &Vecf, f_dims: usize) -> (Vecf, f64) {
	if f_dims == 0 {
		let v = get_gridvalue_n(&pos.as_veci());
		let cv = origin - pos;
		let val = cv ^ &v;
		return (v, val);
	} else {
		let mut fpos = pos.clone();
		let mut cpos = pos.clone();
		fpos[f_dims - 1] = pos[f_dims - 1].floor();
		cpos[f_dims - 1] = pos[f_dims - 1].ceil();

		let (fv, f) = get_perlin_n_grad_inner(&fpos, origin, f_dims - 1);
		let (cv, c) = get_perlin_n_grad_inner(&cpos, origin, f_dims - 1);

		let interp = smoothstep(pos[f_dims - 1] - pos[f_dims - 1].floor());
		let d_interp = d_smoothstep(pos[f_dims - 1] - pos[f_dims - 1].floor());

		let val = f + (c - f) * interp;

		let mut total = (&cv - &fv) * interp + fv;
		total[f_dims - 1] += (c - f) * d_interp;

		return (total, val);
	}
}

fn get_perlin_grad_n(pos: &Vecf) -> (Vecf, f64) {
	let (vec, val) = get_perlin_n_grad_inner(pos, pos, pos.n);

	return (!vec, val);
}

fn render_2d(pixels: Vec<Vec<f64>>) -> Vec<u8> {
	return pixels.iter().flatten().collect::<Vec<&f64>>().iter().map(|x| (*x * 255.0) as u8).collect();
}

fn grey_to_rgb(pixels: Vec<u8>) -> Vec<u8> {
	return pixels.iter().map(|v| vec![*v, *v, *v]).collect::<Vec<Vec<u8>>>().iter().flatten().collect::<Vec<&u8>>().iter().map(|v| **v).collect::<Vec<u8>>();
}

fn vecf_to_rgb(pixels: Vec<Vecf>) -> Vec<u8> {
	return pixels.iter().map(|v| vec![(v[0] * 255.0) as u8, (v[1] * 255.0) as u8, (v[2] * 255.0) as u8]).collect::<Vec<Vec<u8>>>().iter().flatten().collect::<Vec<&u8>>().iter().map(|v| **v).collect::<Vec<u8>>();
}

fn render_3d(grid_size: usize, scale: f64, threshold: f64, w1: f64, w2: f64) -> Vec<u8> {
	let mut pixels = vec![vec![vec![0.0, 0.0, 0.3]; grid_size]; grid_size];
	let mut depth_buffer = vec![vec![false; grid_size]; grid_size];

	for i in (0..grid_size).rev() {
		println!("Rendering slice {i}/{grid_size}");
		for j in (0..grid_size).rev() {
			for k in 0..grid_size {
				let gy = grid_size - 1 - (j * 7 / 10 + k * 3 / 10);
				let gx = i * 7 / 10 + k * 3 / 10;
				let pi = &Veci::new(vec![i as i64, j as i64, k as i64]);
				let p = &Vecf::new(vec![i as f64 / scale, j as f64 / scale, k as f64 / scale, w1, w2]);
				if !depth_buffer[gy][gx] {
					let (norm, val) = get_perlin_grad_n(p);
					if val < threshold {
						let n = 
						if pi.v.iter().any(|a| *a == 0 || *a == (grid_size - 1) as i64) {
							Vecf::new(pi.v.iter().map(|a| if (*a == 0) {-1.0} else if (*a == (grid_size as i64 - 1)) {1.0} else {0.0}).collect())
						} else {
							norm
						};
	
						// pixels[gx][gy][2] = (-t[2]).max(0.0);
						depth_buffer[gy][gx] = true;
	
						pixels[gy][gx][0] = n[0] * 0.5 + 0.5;
						pixels[gy][gx][1] = n[1] * 0.5 + 0.5;
						pixels[gy][gx][2] = n[2] * 0.5 + 0.5;
					}
				}
			}
		}
	}

	return pixels.iter().flatten().collect::<Vec<&Vec<f64>>>().iter().map(|v| *v).flatten().collect::<Vec<_>>().iter().map(|x| (*x * 255.0) as u8).collect();
}

//main image, gradient map
fn render_2d_in_4d(grid_size: usize, scale: f64, z: f64, w: f64) -> Vec<u8> {
    let pixels: Vec<Vec<f64>> = (0..grid_size).map(|y| (0..grid_size).map(|x| (get_perlin_n(&Vecf::new(vec![x as f64 / scale, y as f64 / scale, z as f64, w as f64])) * 0.75 + 0.5)).collect::<Vec<f64>>()).collect::<Vec<_>>();
    let buffer = grey_to_rgb(render_2d(pixels));

    return buffer;
}

fn main() {

	let grid_size = 128;
	let scale = 32.0;

	// for i in 0..101 {
	// 	let buffer = render_3d(grid_size, scale, (i - 50) as f64 * 0.015);

	// 	let error = image::save_buffer(&Path::new(&format!("images/image{i}.png")), &buffer, grid_size as u32, grid_size as u32, image::ExtendedColorType::Rgb8);
	// 	if error.is_err() {
	// 		println!("Fail :(");
	// 	} else {
	// 		println!("Finished image{i}.png");
	// 	}
	// }

	// for i in 0..100 {
	// 	let a = i as f64 / 100.0 * PI * 2.0;
	// 	let buffer = render_2d_in_4d(grid_size, scale, a.cos() * 2.0, a.sin() * 2.0);

	// 	let error = image::save_buffer(&Path::new(&format!("images/image{i}.png")), &buffer, grid_size as u32, grid_size as u32, image::ExtendedColorType::Rgb8);
	// 	if error.is_err() {
	// 		println!("Fail :(");
	// 	} else {
	// 		println!("Finished image{i}.png");
	// 	}
	// }

	let start = SystemTime::now().duration_since(UNIX_EPOCH).expect("time backwards?").as_millis();

	let mut accum = 0;
	for i in 0..1000000 {
		// get_perlin_grad_n(&Vecf::new(vec![i as f64, 0.5, 0.5, 0.5, 0.5]));
		// accum += get_gridvalue_n(&Veci::new(vec![i, 0, 0, 0, 0]))[0];
		// accum += int_vec_hash(&Veci::new(vec![i, 0, 0, 0, 0]), i);
		accum += &Veci::new(vec![i, 0, 0, 0, 0]).v[0];
	}

	let end = SystemTime::now().duration_since(UNIX_EPOCH).expect("time backwards?").as_millis();

	let len = end - start;

	println!("Total time: {}ms, Time per iteration: {}us/{}ns, accum: {}", len, len / 1000, len, accum);

	// for i in 0..16 {
	// 	let a = i as f64 / 16.0 * PI * 2.0;
	// 	let buffer = render_3d(grid_size, scale, -0.2, a.cos() * 0.1, a.sin() * 0.1);

	// 	let error = image::save_buffer(&Path::new(&format!("images/image{i}.png")), &buffer, grid_size as u32, grid_size as u32, image::ExtendedColorType::Rgb8);
	// 	if error.is_err() {
	// 		println!("Fail :(");
	// 	} else {
	// 		println!("Finished image{i}.png");
	// 	}
	// }

}
