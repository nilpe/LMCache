use pyo3::prelude::*;
use rayon::prelude::*;
use std::fs::File;
use std::io::Read;
use std::os::unix::fs::OpenOptionsExt;

/// FSDAX: 複数ファイルを rayon で並列に読み出し、dst バッファに書き込む
///
/// paths: ファイルパスのリスト
/// dst_ptrs: 各チャンクの destination address (pinned DRAM の data_ptr)
/// sizes: 各チャンクのサイズ
/// num_threads: rayon スレッドプールのスレッド数
#[pyfunction]
fn parallel_read_files(
    py: Python<'_>,
    paths: Vec<String>,
    dst_ptrs: Vec<usize>,
    sizes: Vec<usize>,
    num_threads: usize,
) -> PyResult<()> {
    py.allow_threads(|| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        pool.install(|| {
            paths
                .par_iter()
                .zip(dst_ptrs.par_iter())
                .zip(sizes.par_iter())
                .for_each(|((path, &dst_ptr), &size)| {
                    let mut file = File::open(path).unwrap();
                    let dst =
                        unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u8, size) };
                    file.read_exact(dst).unwrap();
                });
        });
    });
    Ok(())
}

/// DevDAX: mmap 済み領域から複数チャンクを rayon で並列にコピー
///
/// base_ptr: DevDAX mmap の base address
/// offsets: 各チャンクの offset (base からの)
/// dst_ptrs: 各チャンクの destination address
/// sizes: 各チャンクのサイズ
/// num_threads: rayon スレッドプールのスレッド数
#[pyfunction]
fn parallel_devdax_copy(
    py: Python<'_>,
    base_ptr: usize,
    offsets: Vec<i64>,
    dst_ptrs: Vec<usize>,
    sizes: Vec<usize>,
    num_threads: usize,
) -> PyResult<()> {
    py.allow_threads(|| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        pool.install(|| {
            offsets
                .par_iter()
                .zip(dst_ptrs.par_iter())
                .zip(sizes.par_iter())
                .for_each(|((&offset, &dst_ptr), &size)| {
                    let src = (base_ptr as isize + offset as isize) as *const u8;
                    let dst = dst_ptr as *mut u8;
                    unsafe {
                        std::ptr::copy_nonoverlapping(src, dst, size);
                    }
                });
        });
    });
    Ok(())
}

#[pymodule]
fn lmcache_fast_read(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parallel_read_files, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_devdax_copy, m)?)?;
    Ok(())
}
