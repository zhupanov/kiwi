import time
import pathlib
import argparse
import multiprocessing
from typing import Dict, List, Optional, Tuple

import cv2
import numpy
import piexif
from PIL import Image, ImageChops, ImageOps, ImageFilter

JPEG_EXTENSIONS = (
    '.JPEG',
    '.JPG'
)  # valid extensions of JPEG images

VIDEO_EXTENSIONS = (
    '.MP4',
)  # valid extensions of video files

DELTA: float = 4.0  # maximum seconds between consecutive images in the same group

CLASSES = (
    'basic',
    'channels',
    'edge',
    'eval',
    'mirror',
    'original',
    'usm',
)

# Grouping ================================================================================

group_count = 0
videos_count = 0


def mod_time(f: pathlib.Path) -> float:
    return f.lstat().st_mtime


def partition_into_groups(file_names: List[pathlib.Path]) -> List[List[pathlib.Path]]:
    groups_list: List[List[pathlib.Path]] = [[file_names[0]]]
    current_group: int = 0
    for file in file_names[1:]:
        if abs(mod_time(file) - mod_time(groups_list[current_group][-1])) <= DELTA:
            groups_list[current_group].append(file)
        else:
            current_group += 1
            groups_list.append([file])
    return groups_list


# Processing a Group ======================================================================


def save(image: Image, base: pathlib.Path, stem: str, kind: str, first_path: Optional[pathlib.Path] = None) -> None:
    out_path = base.joinpath(stem + '_' + kind + '.JPG')
    ImageOps.autocontrast(image, cutoff=0.1).save(out_path)
    if first_path is not None:
        piexif.transplant(str(first_path), str(out_path))  # use EXIF from first_path image


def save_mono(image: Image, base: pathlib.Path, stem: str, kind: str, first_path: Optional[pathlib.Path]) -> None:
    save(ImageOps.grayscale(image), base, stem, kind + '_bw', first_path)


def save_color_and_mono(image: Image, base: pathlib.Path, stem: str, kind: str, first_path: Optional[pathlib.Path]) -> None:
    save(image, base, stem, kind, first_path)
    save_mono(image, base, stem, kind, first_path)


def gen_average(images: List[Image.Image]) -> Image:
    result: Image = images[0]
    for i, image in enumerate(images[1:]):
        alpha: float = 1.0 / float(i+2)
        result = ImageChops.blend(result, image, alpha)
    return result


def gen_darker(images: List[Image.Image]) -> Image:
    result: Image = images[0]
    for image in images[1:]:
        result = ImageChops.darker(result, image)
    return result


def gen_lighter(images: List[Image.Image]) -> Image:
    result: Image = images[0]
    for image in images[1:]:
        result = ImageChops.lighter(result, image)
    return result


def gen_haloed(image: Image, radius: int) -> Image:
    blurred: Image = image.filter(ImageFilter.GaussianBlur(radius))
    result: Image = ImageChops.subtract(image, blurred)
    blurred.close()
    return result


def gen_usm(image: Image, percent: int, radius: int, threshold: int, iterations: int) -> Image:
    result: Image = image
    for _ in range(iterations):
        result = result.filter(ImageFilter.UnsharpMask(percent=percent, radius=radius, threshold=threshold))
    return result


def gen_and_save_basic(first_path: Optional[pathlib.Path],
                       stem: str,
                       first: Image,
                       next_group_first: Image,
                       output_dirs: Dict[str, pathlib.Path]) -> None:
    first_images: List[Image] = [first, next_group_first]
    average2: Image = gen_average(first_images)
    darker2: Image = gen_darker(first_images)
    lighter2: Image = gen_lighter(first_images)
    difference2: Image = ImageChops.difference(first, next_group_first)
    subtract2: Image = ImageChops.subtract(first, next_group_first)
    l_sub_d2: Image = ImageChops.subtract(lighter2, darker2)
    save(average2, output_dirs['basic'], stem, 'average2', first_path)
    save(darker2, output_dirs['basic'], stem, 'darker2', first_path)
    save(lighter2, output_dirs['basic'], stem, 'lighter2', first_path)
    save(difference2, output_dirs['basic'], stem, 'difference2', first_path)
    save(subtract2, output_dirs['basic'], stem, 'subtract2', first_path)
    save(l_sub_d2, output_dirs['basic'], stem, 'l_sub_d2', first_path)


def save_sampled_frames(first_path: Optional[pathlib.Path],
                        stem: str,
                        first: Image,
                        middle: Image,
                        last: Image,
                        output_dirs: Dict[str, pathlib.Path]) -> None:
    save(first, output_dirs['original'], stem, '1_first', first_path)
    if middle != first:
        save(middle, output_dirs['original'], stem, '2_middle', first_path)
    if last != first:
        save(last, output_dirs['original'], stem, '3_last', first_path)


def gen_and_save_mirror(first_path: Optional[pathlib.Path],
                        stem: str,
                        first: Image,
                        middle: Image,
                        last: Image,
                        len_images: int,
                        output_dirs: Dict[str, pathlib.Path]) -> None:
    flipped: Image = ImageOps.flip(last)
    mirrored: Image = ImageOps.mirror(middle)
    flipped_and_mirrored: Image = ImageOps.mirror(flipped)

    two_combo: Image = ImageChops.subtract(first, flipped)
    save(two_combo, output_dirs['mirror'], stem, 'combo_2', first_path)

    three_combo: Image = ImageChops.blend(two_combo, mirrored, 0.33)
    save(three_combo, output_dirs['mirror'], stem, 'combo_3', first_path)

    four_combo: Image = ImageChops.blend(three_combo, flipped_and_mirrored, 0.25)
    save(four_combo, output_dirs['mirror'], stem, 'combo_4', first_path)

    four_combo_2: Image = ImageOps.flip(ImageChops.difference(
        ImageChops.screen(ImageChops.difference(first, mirrored), flipped),
        flipped_and_mirrored
    ))
    save(four_combo_2, output_dirs['mirror'], stem, 'combo_4_2', first_path)

    if len_images > 1:
        pairwise_subtract: Image = ImageChops.subtract(
            ImageChops.blend(first, flipped, 0.5),
            ImageChops.blend(mirrored, flipped_and_mirrored, 0.5))
        save(pairwise_subtract, output_dirs['mirror'], stem, 'pairwise_sub', first_path)

        sub_inv: Image = ImageChops.subtract(first, ImageChops.invert(flipped))
        save(sub_inv, output_dirs['mirror'], stem, 'inv_flip_sub', first_path)


def gen_and_save_edge(first_path: Optional[pathlib.Path],
                      stem: str,
                      first: Image,
                      output_dirs: Dict[str, pathlib.Path]) -> None:
    offset: Image = ImageChops.offset(first, 1)
    edge: Image = ImageChops.subtract(first, offset)
    offset.close()
    save(edge, output_dirs['edge'], stem, 'edge_1', first_path)


def gen_and_save_usm(first_path: Optional[pathlib.Path],
                     stem: str,
                     first: Image,
                     r_diff_g: Image,
                     output_dirs: Dict[str, pathlib.Path]) -> None:
    halo_150: Image = gen_haloed(first, 150)
    radius: int = int(20 * first.width / 1400.0)
    save(gen_usm(r_diff_g, 500, radius, 1, 5), output_dirs['usm'],
         stem, 'usm_%s_%d_%03d_%02d_%02d' % ('r_diff_g', 500, radius, 1, 5), first_path)
    save(gen_usm(halo_150, 500, radius, 1, 5), output_dirs['usm'],
         stem, 'usm_%s_%d_%03d_%02d_%02d' % ('halo', 500, radius, 1, 5), first_path)


def gen_and_save_eval(first_path: Optional[pathlib.Path],
                      stem: str,
                      first: Image,
                      darker: Image,
                      lighter: Image,
                      subtract: Image,
                      len_images: int,
                      output_dirs: Dict[str, pathlib.Path]) -> None:
    def f1(x: int) -> int:
        return round((x - 127) * (x - 127) / 64)

    save(Image.eval(first, f1), output_dirs['eval'], stem, 'eval_first_sqr', first_path)
    if len_images > 1:
        save(Image.eval(darker, f1), output_dirs['eval'], stem, 'eval_darker_sqr', first_path)
        save(Image.eval(lighter, f1), output_dirs['eval'], stem, 'eval_lighter_sqr', first_path)
        save(Image.eval(subtract, f1), output_dirs['eval'], stem, 'eval_subtract_sqr', first_path)

    def f2(x: int) -> int:
        if x < 85:
            return 3 * x
        if x < 170:
            return (-3) * (x - 170)
        return 3 * (x - 170)

    save(Image.eval(first, f2), output_dirs['eval'], stem, 'eval_first_zigzag', first_path)
    if len_images > 1:
        save(Image.eval(darker, f2), output_dirs['eval'], stem, 'eval_darker_zigzag', first_path)
        save(Image.eval(lighter, f2), output_dirs['eval'], stem, 'eval_lighter_zigzag', first_path)
        save(Image.eval(subtract, f2), output_dirs['eval'], stem, 'eval_subtract_zigzag', first_path)


# images: list of Image; next_group_image: Image
def combine_images(args: argparse.Namespace,
                   images: List[Image.Image],
                   next_group_first: Image,
                   output_dirs: Dict[str, pathlib.Path],
                   stem: str,
                   first_path: Optional[pathlib.Path] = None) -> None:
    first: Image = images[0]
    middle: Image = images[len(images) // 2]
    last: Image = images[-1]

    save_sampled_frames(first_path, stem, first, middle, last, output_dirs)

    average: Image = gen_average(images)
    darker: Image = gen_darker(images)
    lighter: Image = gen_lighter(images)
    difference: Image = ImageChops.difference(first, last)
    subtract: Image = ImageChops.subtract(first, last)
    l_sub_d: Image = ImageChops.subtract(lighter, darker)
    if args.basic and len(images) > 1:
        save(average, output_dirs['basic'], stem, 'average', first_path)
        save(darker, output_dirs['basic'], stem, 'darker', first_path)
        save(lighter, output_dirs['basic'], stem, 'lighter', first_path)
        save(difference, output_dirs['basic'], stem, 'difference', first_path)
        save(subtract, output_dirs['basic'], stem, 'subtract', first_path)
        save(l_sub_d, output_dirs['basic'], stem, 'l_sub_d', first_path)

    if args.basic:
        gen_and_save_basic(first_path, stem, first, next_group_first, output_dirs)

    if args.mirror:
        gen_and_save_mirror(first_path, stem, first, middle, last, len(images), output_dirs)

    if args.edge:
        gen_and_save_edge(first_path, stem, first, output_dirs)

    r, g, _ = first.split()
    r_diff_g: Image = ImageChops.difference(r, g)
    if args.channels:
        save(r_diff_g, output_dirs['channels'], stem, 'r_dif_g', first_path)

    if args.usm:
        gen_and_save_usm(first_path, stem, first, r_diff_g, output_dirs)

    if args.eval:
        gen_and_save_eval(first_path, stem, first, darker, lighter, subtract, len(images), output_dirs)


def process_group(i: int,
                  groups_list: List[List[pathlib.Path]],
                  args: argparse.Namespace,
                  output_dirs: Dict[str, pathlib.Path]) -> Tuple[int, int, int, Optional[Exception]]:
    start: float = time.time()
    group: List[pathlib.Path] = groups_list[i]
    next_group: List[pathlib.Path] = groups_list[i+1] if (i < len(groups_list) - 1) else groups_list[0]
    try:
        combine_images(
            args=args,
            images=[Image.open(file) for file in group],
            next_group_first=Image.open(next_group[0]),
            output_dirs=output_dirs,
            stem=group[0].stem,
            first_path=group[0])
    except (ValueError, Exception) as e:  # pylint: disable=W0703
        return i, len(group), round(time.time() - start), e
    return i, len(group), round(time.time() - start), None


def numpy_ndarray_to_pillow_image(numpy_ndarray: numpy.ndarray) -> Image:
    return Image.fromarray(cv2.cvtColor(numpy_ndarray, cv2.COLOR_BGR2RGB))  # pylint: disable=E1101


def extract_images_as_pillow_from_video(video: pathlib.Path) -> List[Image.Image]:
    vc = cv2.VideoCapture(str(video))  # pylint: disable=E1101
    try:
        if not vc.isOpened():
            raise RuntimeError(f'Unable to open {str(video)}')
        pillow_images: List[Image] = list()
        frame_count: int = int(vc.get(cv2.CAP_PROP_FRAME_COUNT)) - 1  # pylint: disable=E1101
        for _ in range(frame_count):
            success, numpy_ndarray = vc.read()
            assert success
            pillow_images.append(numpy_ndarray_to_pillow_image(numpy_ndarray))
        return pillow_images
    finally:
        vc.release()


def extract_first_image_as_pillow_from_video(video: pathlib.Path) -> Image:
    vc = cv2.VideoCapture(str(video))  # pylint: disable=E1101
    try:
        success, numpy_ndarray = vc.read()
        assert success
        return numpy_ndarray_to_pillow_image(numpy_ndarray)
    finally:
        vc.release()


def process_video(i: int,
                  videos: List[pathlib.Path],
                  args: argparse.Namespace,
                  output_dirs: Dict[str, pathlib.Path]) -> Tuple[int, int, int, Optional[Exception]]:
    start: float = time.time()
    video: pathlib.Path = videos[i]
    next_video: pathlib.Path = videos[i+1] if i < len(videos) - 1 else videos[0]
    images = list()
    try:
        images = extract_images_as_pillow_from_video(video)
        if args.max_frames < len(images):
            images = [img for (i, img) in enumerate(images)
                      if i % (round(len(images) / args.max_frames)) == 0][0:args.max_frames]
        combine_images(
            args=args,
            images=images,
            next_group_first=extract_first_image_as_pillow_from_video(next_video),
            output_dirs=output_dirs,
            stem=video.stem)

    except (ValueError, Exception) as e:  # pylint: disable=W0703
        return i, len(images), round(time.time() - start), e

    return i, len(images), round(time.time() - start), None


# Parallel Processing Machinery =============================================================


def report_group(x: Tuple[int, int, int, Optional[Exception]]) -> None:
    global group_count
    group_count -= 1
    i, n, duration, e = x
    print(f'Processed group {i:3d} of size {n:2d} in {duration:2d} seconds: {"SUCCESS" if not e else str(e)};  '
          f'remaining groups count: {group_count}')


def process_groups_in_parallel(cores: int,
                               groups_list: List[List[pathlib.Path]],
                               args: argparse.Namespace,
                               output_dirs: Dict[str, pathlib.Path]) -> None:
    pool: multiprocessing.pool.Pool = multiprocessing.Pool(cores)
    for i in range(len(groups_list)):
        pool.apply_async(process_group, (i, groups_list, args, output_dirs), callback=report_group)
    pool.close()
    pool.join()


def report_video(x: Tuple[int, int, int, Optional[Exception]]) -> None:
    global videos_count
    videos_count -= 1
    i, n, duration, e = x  # i: video index number; n: video frames count; e: exception (or None)
    print('Processed video %3d, extracting %2d frames, in %2d seconds: %s;  remaining videos count: %d' %
          (i, n, duration, 'SUCCESS' if not e else str(e), videos_count))


def process_videos_in_parallel(cores: int,
                               videos_list: List[pathlib.Path],
                               args: argparse.Namespace,
                               output_dirs: Dict[str, pathlib.Path]) -> None:
    pool: multiprocessing.pool.Pool = multiprocessing.Pool(cores)
    for i in range(len(videos_list)):
        pool.apply_async(process_video, (i, videos_list, args, output_dirs), callback=report_video)
    pool.close()
    pool.join()


# Main Auxiliaries ==========================================================================


def parse_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description='Combine groups of images for Multiple Exposure techniques.')
    parser.add_argument('-i', '--in', dest='input_dir', action='store', required=True,
                        help='root directory containing input groups of images')
    parser.add_argument('-o', '--out', dest='output_dir', action='store', required=True,
                        help='output directory to store the combined images in')
    parser.add_argument('--basic', dest='basic', action='store_true', help='generate basic variants', default=False)
    parser.add_argument('--mirror', dest='mirror', action='store_true', help='generate mirror variants', default=False)
    parser.add_argument('--edge', dest='edge', action='store_true', help='generate edge variants', default=False)
    parser.add_argument('--usm', dest='usm', action='store_true', help='generate usm variants', default=False)
    parser.add_argument('--channels', dest='channels', action='store_true',
                        help='generate channels variants', default=False)
    parser.add_argument('--eval', dest='eval', action='store_true', help='generate eval variants', default=False)
    parser.add_argument('--all', dest='all', action='store_true', help='generate all variants', default=False)

    parser.add_argument('--max_frames', action="store", dest="max_frames", type=int,
                        help='max frames (>= 1) to extract from each video', default=100)

    args: argparse.Namespace = parser.parse_args()
    args.original = True  # we always generate "original", unconditionally
    if args.all:
        args.basic = True
        args.mirror = True
        args.edge = True
        args.eval = True
        args.usm = True
        args.channels = True
    assert args.max_frames > 0
    return args


def create_output_dirs(args: argparse.Namespace) -> Dict[str, pathlib.Path]:
    output_path: pathlib.Path = pathlib.Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    output_dirs: Dict[str, pathlib.Path] = dict()
    for c in CLASSES:
        if args.__getattribute__(c):
            output_dirs[c] = output_path.joinpath(c)
            output_dirs[c].mkdir(exist_ok=True)
    return output_dirs


# Main ======================================================================================


def main() -> None:
    start: float = time.time()
    args: argparse.Namespace = parse_args()
    base = pathlib.Path(args.input_dir)
    file_names: List[pathlib.Path] = [
        pth for pth in base.glob('**/*.*') if pth.is_file() and pth.suffix.upper() in JPEG_EXTENSIONS]
    video_names: List[pathlib.Path] = [
        pth for pth in base.glob('**/*.*') if pth.is_file() and pth.suffix.upper() in VIDEO_EXTENSIONS]
    groups_list: List[List[pathlib.Path]] = partition_into_groups(file_names) if len(file_names) > 0 else list()
    global group_count
    group_count = len(groups_list)
    global videos_count
    videos_count = len(video_names)
    cores: int = multiprocessing.cpu_count() // 2
    if group_count > 0:
        print("Found " + str(len(file_names)) + " files under " + str(base) + " and partitioned them into " +
              str(group_count) + " groups")
        group_output_dirs: Dict[str, pathlib.Path] = create_output_dirs(args)
        print("Requested image classes: " + str(sorted(set(group_output_dirs.keys()))))
        print("Processing using %s slave sub-processes." % cores)
        print('---------------------------------------------------------------------------------------')
        process_groups_in_parallel(cores, groups_list, args, group_output_dirs)
        print('---------------------------------------------------------------------------------------')
        print("Finished processing " + str(len(file_names)) + " images under " + str(base) + ", partitioned into " +
              str(len(groups_list)) + " groups, using " + str(cores) + " parallel processes.")
        print("Results saved to " + args.output_dir)
        print('Total images processing time: %d seconds.' % round(time.time() - start))

    if videos_count > 0:
        start = time.time()
        video_output_dirs: Dict[str, pathlib.Path] = create_output_dirs(args)
        print('=======================================================================================')
        print("Found " + str(len(video_names)) + " videos under " + str(base))
        print("Requested image classes: " + str(sorted(set(video_output_dirs.keys()))))
        print("Processing using %s slave sub-processes." % cores)
        print('---------------------------------------------------------------------------------------')
        process_videos_in_parallel(cores, video_names, args, video_output_dirs)
        print('---------------------------------------------------------------------------------------')
        print("Finished processing " + str(len(video_names)) + " videos under " + str(base) +
              " using " + str(cores) + " parallel processes.")
        print("Results saved to " + args.output_dir)
        print('Total videos processing time: %d seconds.' % round(time.time() - start))

    if group_count == 0 and videos_count == 0:
        print('Failed to find images or videos under %s.  Terminating...' % str(base))


if __name__ == '__main__':
    main()
