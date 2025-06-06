use std::f64::consts::PI;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::ops;
use std::{path::Path};

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
		let nv: Vec<i32> = self.v.iter().map(|a| *a as i32).collect();
		let len = nv.len();
		return Veci {v: nv, n: len};
	}
}

impl ops::Index<usize> for Vecf {
	type Output = f64;

	fn index(&self, index: usize) -> &Self::Output {
		return &self.v[index];
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
	v: Vec<i32>,
	n: usize
}

impl Veci {
	fn new(v: Vec<i32>) -> Veci {
		let len = v.len();
		return Veci {v, n: len};
	}

	fn as_vecf(&self) -> Vecf {
		let nv: Vec<f64> = self.v.iter().map(|a| *a as f64).collect();
		let len = nv.len();
		return Vecf {v: nv, n: len};
	}
}

impl ops::Div<i32> for Veci {
	type Output = Veci;

	fn div(self, o: i32) -> Veci {
		let nv: Vec<i32> = self.v.iter().map(|a| a / o).collect();
		let len = nv.len();
		return Veci {v: nv, n: len};
	}
}

impl ops::Div<i32> for &Veci {
	type Output = Veci;

	fn div(self, o: i32) -> Veci {
		let nv: Vec<i32> = self.v.iter().map(|a| a / o).collect();
		let len = nv.len();
		return Veci {v: nv, n: len};
	}
}

fn smoothstep(x: f64) -> f64 {
	return 3.0 * x * x - 2.0 * x * x * x;
}

fn get_gridvalue_n(pos: Veci) -> Vecf {
	let mut s = DefaultHasher::new();

	let mut bm: Vec<f64> = vec![0.0; pos.n];

	for (i, _) in pos.v.iter().enumerate() {
		(pos.v.clone(), i * 2).hash(&mut s);
		let t = (s.finish() % 256) as f64 / 256.0 * 2.0 * PI;
		(pos.v.clone(), i * 2 + 1).hash(&mut s);
		let r = (((s.finish() % 256 + 1) as f64 / 256.0).ln() * -2.0).sqrt();
		bm[i] = r * t.cos();
	}

	let out = !Vecf::new(bm.clone());

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
		let v = get_gridvalue_n(pos.as_veci());
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

	return val * 0.5 + 0.5;
}

fn get_perlin_n_grad_inner(pos: &Vecf, origin: &Vecf, f_dims: usize) -> Vecf {
	if f_dims == 0 {
		let v = get_gridvalue_n(pos.as_veci());
		return v;
	} else {
		let mut fpos = pos.clone();
		let mut cpos = pos.clone();
		fpos[f_dims - 1] = pos[f_dims - 1].floor();
		cpos[f_dims - 1] = pos[f_dims - 1].ceil();

		let fv = get_perlin_n_grad_inner(&fpos, origin, f_dims - 1);
		let cv = get_perlin_n_grad_inner(&cpos, origin, f_dims - 1);

		let interp = smoothstep(pos[f_dims - 1] - pos[f_dims - 1].floor());
		let total = (&cv - &fv) * interp + fv;

		return total;
	}
}

fn get_perlin_grad_n(pos: &Vecf) -> Vecf {
	let val = !get_perlin_n_grad_inner(pos, pos, pos.n);

	return val;
}

fn render_2d(pixels: Vec<Vec<f64>>) -> Vec<u8> {
	return pixels.iter().flatten().collect::<Vec<&f64>>().iter().map(|x| (*x * 255.0) as u8).collect();
}

fn grey_to_rgb(pixels: Vec<u8>) -> Vec<u8> {
	return pixels.iter().map(|v| vec![*v, *v, *v]).collect::<Vec<Vec<u8>>>().iter().flatten().collect::<Vec<&u8>>().iter().map(|v| **v).collect::<Vec<u8>>();
}

fn render_3d(grid_size: usize, scale: f64) -> Vec<u8> {
	let mut pixels = vec![vec![vec![1.0; 3]; grid_size]; grid_size];
	let mut depth_buffer = vec![vec![false; grid_size]; grid_size];

	for i in (0..grid_size).rev() {
		println!("Rendering slice {i}/{grid_size}");
		for j in (0..grid_size).rev() {
			for k in 0..grid_size {
				let pi = &Veci::new(vec![i as i32, j as i32, k as i32]);
				let p = &(pi.as_vecf() / scale);
				if !depth_buffer[grid_size - 1 - (j * 7 / 10 + k * 3 / 10)][i * 7 / 10 + k * 3 / 10] && get_perlin_n(p) > 0.5 {
					let t = if pi.v.iter().any(|a| *a == 0 || *a == (grid_size - 1) as i32) {
						let mut a = Vecf::new(pi.v.iter().map(|a| if (*a == 0) {-1.0} else if (*a == (grid_size as i32 - 1)) {1.0} else {0.0}).collect());
						a[0] = a[0];
						a[1] = -a[1];
						a
					} else {
						-get_perlin_grad_n(p)
					};

					pixels[grid_size - 1 - (j * 7 / 10 + k * 3 / 10)][i * 7 / 10 + k * 3 / 10][0] = (-t[0]).max(0.0);
					pixels[grid_size - 1 - (j * 7 / 10 + k * 3 / 10)][i * 7 / 10 + k * 3 / 10][1] = (-t[1]).max(0.0);
					pixels[grid_size - 1 - (j * 7 / 10 + k * 3 / 10)][i * 7 / 10 + k * 3 / 10][2] = (-t[2]).max(0.0);
					depth_buffer[grid_size - 1 - (j * 7 / 10 + k * 3 / 10)][i * 7 / 10 + k * 3 / 10] = true;

					// pixels[grid_size - 1 - (j * 7 / 10 + k * 3 / 10)][i * 7 / 10 + k * 3 / 10][0] = voxels[k][j][i];
					// pixels[grid_size - 1 - (j * 7 / 10 + k * 3 / 10)][i * 7 / 10 + k * 3 / 10][1] = voxels[k][j][i];
					// pixels[grid_size - 1 - (j * 7 / 10 + k * 3 / 10)][i * 7 / 10 + k * 3 / 10][2] = voxels[k][j][i];
				}
			}
		}
	}

	return pixels.iter().flatten().collect::<Vec<&Vec<f64>>>().iter().map(|v| *v).flatten().collect::<Vec<_>>().iter().map(|x| (*x * 255.0) as u8).collect();
}

fn main() {

	// let pixels: [[u8; 256]; 256] = [(0..256).map(|y| (get_value(0, y) * 255.0) as u8).collect::<Vec<u8>>().try_into().unwrap(); 256];
	let grid_size = 64;
	let scale = 64.0;
	// let pixels: Vec<Vec<f64>> = (0..grid_size).map(|y| (0..grid_size).map(|x| (get_perlin_n(Vecf::new(vec![x as f64 / scale, y as f64 / scale])))).collect::<Vec<f64>>()).collect::<Vec<_>>();
	// let voxels: Vec<Vec<Vec<f64>>> = (0..grid_size).map(|z| {
	//   println!("z={z} done");

	//   (0..grid_size).map(|y| (0..grid_size).map(|x| (get_perlin_n(Vecf::new(vec![x as f64 / scale, y as f64 / scale, z as f64 / scale])))).collect::<Vec<f64>>()).collect::<Vec<Vec<f64>>>()
	// }).collect::<Vec<_>>();
	println!("Noise generated");
	let buffer = render_3d(grid_size, scale);
	// let buffer = grey_to_rgb(render_2d(pixels));
	
	let error = image::save_buffer(&Path::new("image.png"), &buffer, grid_size as u32, grid_size as u32, image::ExtendedColorType::Rgb8);
	if error.is_err() {
		println!("Fail :(");
	} else {
		println!("Finished :D");
	}

}
