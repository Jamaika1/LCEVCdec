# Copyright (c) V-Nova International Limited 2023-2026. All rights reserved.
# This software is licensed under the BSD-3-Clause-Clear License by V-Nova Limited.
# No patent licenses are granted under this license. For enquiries about patent licenses,
# please contact legal@v-nova.com.
# The LCEVCdec software is a stand-alone project and is NOT A CONTRIBUTION to any other project.
# If the software is incorporated into another project, THE TERMS OF THE BSD-3-CLAUSE-CLEAR LICENSE
# AND THE ADDITIONAL LICENSING INFORMATION CONTAINED IN THIS FILE MUST BE MAINTAINED, AND THE
# SOFTWARE DOES NOT AND MUST NOT ADOPT THE LICENSE OF THE INCORPORATING PROJECT. However, the
# software may be incorporated into a project under a compatible license provided the requirements
# of the BSD-3-Clause-Clear license are respected, and V-Nova Limited remains
# licensor of the software ONLY UNDER the BSD-3-Clause-Clear license (not the compatible license).
# ANY ONWARD DISTRIBUTION, WHETHER STAND-ALONE OR AS PART OF ANY OTHER PROJECT, REMAINS SUBJECT TO
# THE EXCLUSION OF PATENT LICENSES PROVISION OF THE BSD-3-CLAUSE-CLEAR LICENSE.

import os

from utilities.paths import format_path, get_executable
from utilities.runner import get_runner
from utilities.test_runner import BaseTest
from utilities.assets import get_encode
from utilities.config import logger
from utilities.histogram import pixel_deviation_histogram
from utilities.load_tests import csv_json_to_dict
from math import sqrt


class Test(BaseTest):
    def test(self, test, test_dir):
        test['cli']['--base'] = format_path(test['cli']['--base'])
        default_yuv = 'no_dithering'
        dithering_yuv = 'dithering'
        dither_strength = int(test['meta']['dither_strength'])
        json_config = csv_json_to_dict(test.get('json', {}))
        histogram_tolerance = float(test['meta']['tolerance'])
        max_std_dev = float(test['meta']['max_standard_deviation'])

        encode_path = get_encode(test['erp'])
        self.log_erp_command(encode_path)
        runner = get_runner(get_executable('lcevc_dec_test_harness'), test_dir)
        if json_config:
            runner.set_json_param('--configuration', json_config)
        runner.set_params_from_config(test['cli'])
        if test['cli'].get('--base'):
            runner.set_param('--base', format_path(test['cli']['--base']), path=True)
            runner.set_param('--lcevc', encode_path, path=True)
        else:
            runner.set_param('--input', encode_path, path=True)
        runner.set_param('--output', default_yuv)

        self.record_cmd('no_dither', runner.get_command_line(as_string=True))
        logger.debug(
            f"Running dec harness (no dithering): {runner.get_command_line(as_string=True)}")
        runner.run(assert_rc=True)

        json_config['dither_strength'] = dither_strength
        json_config['allow_dithering'] = True
        runner.set_json_param('--configuration', json_config)
        runner.set_param('--output', dithering_yuv)
        logger.debug(f"Running dec harness (dithering): {runner.get_command_line(as_string=True)}")
        self.record_cmd('with_dither', runner.get_command_line(as_string=True))
        runner.run(assert_rc=True)
        if test['meta']['color_space'] == 'yuv':
            subsampling_ratio = test['meta']['subsampling_ratio']
            filename_extension = '_' + test['meta']['width'] + 'x' + test['meta']['height'] + \
                ('' if test['meta']['bit_depth'] == '8' else '_' + test['meta']['bit_depth'] + 'bit') + \
                '_p' + subsampling_ratio + '.' + test['meta']['color_space']
        else:
            subsampling_ratio = '420'
            filename_extension = '_' + test['meta']['width'] + 'x' + \
                test['meta']['height'] + '.' + test['meta']['color_space']

        ref_path = os.path.join(test_dir, default_yuv + filename_extension)
        dither_path = os.path.join(test_dir, dithering_yuv + filename_extension)
        y_histo, u_histo, v_histo = pixel_deviation_histogram(ref_path,
                                                              dither_path,
                                                              int(test['meta']['bit_depth']),
                                                              int(test['meta']['width']),
                                                              int(test['meta']['height']),
                                                              subsampling_ratio,
                                                              test['meta']['color_space'])

        if test['erp'].get('--encoding_plane_mode', 'y') == 'y':
            histos = {'Y': y_histo}
            for plane, histo in {'U': u_histo, 'V': v_histo}.items():
                assert len(histo) == 1, f"Un-enhanced {plane} plane has dithering but it shouldn't"
        else:  # YUV chroma residuals
            histos = {'Y': y_histo, 'U': u_histo, 'V': v_histo}

        num_buckets = 2 * dither_strength + 1
        expected_bucket_size = 100.0 / num_buckets
        low_gate = (1 - histogram_tolerance) * expected_bucket_size
        high_gate = (1 + histogram_tolerance) * expected_bucket_size
        sum_squared = 0

        for plane, histo in histos.items():
            assert len(histo) == num_buckets, f"{plane} plane expected {num_buckets} " \
                f"dithering buckets and received {len(histo)}"

            for bucket, size in histo.items():
                assert -dither_strength <= bucket <= dither_strength, \
                    f"Unexpected dither value {bucket} for strength {dither_strength} in {plane} plane"
                assert low_gate < size < \
                       high_gate, f"{plane} plane bucket value {bucket} has size {size}, " \
                                  f"expected tolerance range {low_gate} - {high_gate}"

                sum_squared += pow(size - expected_bucket_size, 2)

            std_dev = sqrt(sum_squared / 100.0)
            assert std_dev < max_std_dev, f"{plane} plane standard deviation {std_dev} exceeds maximum allowed value {max_std_dev}"
