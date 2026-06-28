import os
import dynamic_panoramator
import time

def main():
  experiments = ['boat.mp4']

  for experiment in experiments:
    exp_no_ext = experiment.split('.')[0]
    os.system('mkdir dump')
    os.system('mkdir dump/%s' % exp_no_ext)
    os.system('ffmpeg -i videos/%s dump/%s/%s%%03d.jpg' % (experiment, exp_no_ext, exp_no_ext))

    s = time.time()
    n_frames = len([f for f in os.listdir('dump/%s' % exp_no_ext) if f.endswith('.jpg')])
    panorama_generator = dynamic_panoramator.PanoramicVideoGenerator('dump/%s/' % exp_no_ext, exp_no_ext, n_frames)
    panorama_generator.align_images(translation_only=True)
    panorama_generator.generate_panoramic_images(9)
    print(' time for %s: %.1f' % (exp_no_ext, time.time() - s))

    panorama_generator.save_panoramas_to_video()


if __name__ == '__main__':
  main()
