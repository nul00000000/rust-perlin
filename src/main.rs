use std::f64::consts::PI;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::ops;
use std::{path::Path};

struct Vec2 {
    x: f64,
    y: f64,
}

impl ops::Add for Vec2 {
    type Output = Vec2;

    fn add(self, o: Vec2) -> Vec2 {
        return Vec2 {x: self.x + o.x, y: self.y + o.y};
    }
}

impl ops::Mul for Vec2 {
    type Output = f64;

    fn mul(self, o: Vec2) -> f64 {
        return self.x * o.x + self.y * o.y;
    }
}

fn vec2(angle: f64) -> Vec2 {
    return Vec2 {x: angle.cos(), y: angle.sin()};
}

fn smoothstep(x: f64) -> f64 {
    return 3.0 * x * x - 2.0 * x * x * x;
}

fn get_gridvalue(x: i32, y: i32) -> f64 {
    let mut s = DefaultHasher::new();
    (x, y).hash(&mut s);
    let num = ((s.finish() % 256) as f64);
    return num / 255.0;
}

fn get_value(x: f64, y: f64) -> f64 {
    let ff = get_gridvalue(x.floor() as i32, y.floor() as i32);
    let fc = get_gridvalue(x.floor() as i32, y.floor() as i32 + 1);
    let cf = get_gridvalue(x.floor() as i32 + 1, y.floor() as i32);
    let cc = get_gridvalue(x.floor() as i32 + 1, y.floor() as i32 + 1);

    let h1 = x - x.floor();
    let v1 = y - y.floor();

    let h = smoothstep(h1);
    let v = smoothstep(v1);

    let top = ff + (cf - ff) * h;
    let bot = fc + (cc - fc) * h;
    let total = top + (bot - top) * v;

    return total;
}

fn get_perlin(x: f64, y: f64) -> f64 {
    let ffv = vec2(get_gridvalue(x.floor() as i32, y.floor() as i32) * PI * 2.0);
    let fcv = vec2(get_gridvalue(x.floor() as i32, y.floor() as i32 + 1) * PI * 2.0);
    let cfv = vec2(get_gridvalue(x.floor() as i32 + 1, y.floor() as i32) * PI * 2.0);
    let ccv = vec2(get_gridvalue(x.floor() as i32 + 1, y.floor() as i32 + 1) * PI * 2.0);

    let h1 = x - x.floor();
    let v1 = y - y.floor();

    let ff = Vec2 {x: h1, y: v1} * ffv;
    let fc = Vec2 {x: h1, y: v1 - 1.0} * fcv;
    let cf = Vec2 {x: h1 - 1.0, y: v1} * cfv;
    let cc = Vec2 {x: h1 - 1.0, y: v1 - 1.0} * ccv;

    let h = smoothstep(h1);
    let v = smoothstep(v1);

    let top = ff + (cf - ff) * h;
    let bot = fc + (cc - fc) * h;
    let total = top + (bot - top) * v;

    return total * 0.5 + 0.5;
}

fn main() {

    // let pixels: [[u8; 256]; 256] = [(0..256).map(|y| (get_value(0, y) * 255.0) as u8).collect::<Vec<u8>>().try_into().unwrap(); 256];
    let pixels: Vec<Vec<f64>> = (0..=255).map(|y| (0..256).map(|x| (get_perlin(x as f64 / 16.0, y as f64 / 16.0))).collect::<Vec<f64>>()).collect::<Vec<_>>();
    let buffer: Vec<u8> = pixels.iter().flatten().collect::<Vec<_>>().iter().map(|x| (*x * 255.0) as u8).collect();
    
    let error = image::save_buffer(&Path::new("image.png"), &buffer, 256, 256, image::ExtendedColorType::L8);
    if error.is_err() {
        println!("Fail :(");
    } else {
        println!("Finished :D");
    }

}
