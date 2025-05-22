# app.py
from flask import Flask, request, jsonify
from decimal import Decimal
from datetime import datetime, timedelta
import pytz
import logging

app = Flask(__name__)
logging.basicConfig(filename='cu_time_converter_v1_0_9.log', level=logging.DEBUG)

# Constants (v1.0.9) -- must be consistent with the converter.py versions
BASE_CU = Decimal('3079913911800.94954834')
CU_OFFSET = Decimal('335739.82')
SECONDS_PER_YEAR = Decimal('31557600')
COSMIC_LIFESPAN = Decimal('13.8e9')
NASA_LIFESPAN = Decimal('13.797e9')
CONVERGENCE_YEAR = Decimal('2029')
# For CU-Time branch (cosmic):
BASE_DATE_UTC = datetime(2025, 5, 16, 18, 30, 0, tzinfo=pytz.UTC)
NASA_REFERENCE = Decimal('13797000000')

# NEW: Define a dedicated base date for NASA branch conversions.
# This is chosen so that:
#   BASE_DATE_NASA − ((NASA_REFERENCE − input) × SECONDS_PER_YEAR)
# yields approximately "2007-10-15 00:00:00 UTC" when input = 13796999982.79
# In our final converter.py, we used:
#   BASE_DATE_NASA = datetime(2024, 12, 30, 4, 48, 0, tzinfo=pytz.UTC)
BASE_DATE_NASA = datetime(2024, 12, 30, 4, 48, 0, tzinfo=pytz.UTC)


@app.route('/cu_to_gregorian', methods=['POST'])
def cu_to_gregorian_api():
    try:
        data = request.json
        cu_time = Decimal(str(data.get('cu_time')))
        timezone = data.get('timezone', 'UTC')
        scales = data.get('scales', [])
        calibration = data.get('calibration', 'cu')
        compression_model = data.get('compression_model', 'logarithmic')

        is_nasa = calibration.lower() == 'nasa' or (cu_time >= Decimal('1e9') and cu_time < Decimal('1e11'))
        if is_nasa:
            # Use the NASA base date!
            years_from_reference = NASA_REFERENCE - cu_time
            total_seconds = years_from_reference * SECONDS_PER_YEAR
            gregorian_time = BASE_DATE_NASA - timedelta(seconds=float(total_seconds))
            cu_decimal = cu_time * ((BASE_CU + CU_OFFSET) / NASA_REFERENCE)
            nasa_time = cu_time  # remains as given
        else:
            ratio = COSMIC_LIFESPAN / CONVERGENCE_YEAR
            cu_diff = cu_time - (BASE_CU + CU_OFFSET)
            gregorian_seconds = (cu_diff * SECONDS_PER_YEAR) / ratio
            if compression_model.lower() == 'polynomial' and gregorian_seconds < Decimal(-31557600) * Decimal('1000'):
                gregorian_seconds *= Decimal('1.0001')
            gregorian_time = BASE_DATE_UTC + timedelta(seconds=int(gregorian_seconds))
            cu_decimal = cu_time
            nasa_time = cu_decimal * Decimal('0.9997826')

        tz = pytz.timezone(timezone) if timezone != 'UTC' else pytz.UTC
        output = {
            'gregorian': gregorian_time.astimezone(tz).strftime('%Y-%m-%d %H:%M:%S %Z'),
            'cu_time': f'{cu_decimal:.2f}',
            'nasa_time': f'{nasa_time:.2f}'
        }
        if 'geological' in scales:
            output['geological'] = 'Holocene'
        if 'cosmic' in scales:
            output['cosmic'] = 'Dec-31'
        logging.info(f"Converted {cu_time} to {output['gregorian']}")
        return jsonify(output)
    except Exception as e:
        logging.error(f"Error in cu_to_gregorian: {str(e)}")
        return jsonify({'error': str(e)}), 400


@app.route('/gregorian_to_cu', methods=['POST'])
def gregorian_to_cu_api():
    try:
        data = request.json
        gregorian_time_str = data.get('gregorian_time')
        timezone = data.get('timezone', 'UTC')
        lifespan = data.get('lifespan', 'CU')
        compression_model = data.get('compression_model', 'logarithmic')

        # We assume ISO format for the Gregorian time including timezone offset
        parsed_time = datetime.strptime(gregorian_time_str, '%Y-%m-%dT%H:%M:%S%z')

        if lifespan.upper() == "NASA":
            time_diff = parsed_time - BASE_DATE_NASA
        else:
            time_diff = parsed_time - BASE_DATE_UTC

        total_seconds = Decimal(time_diff.total_seconds())

        if lifespan.upper() == "NASA":
            years_from_reference = -total_seconds / SECONDS_PER_YEAR
            nasa_time = NASA_REFERENCE - years_from_reference
            cu_time = nasa_time * ((BASE_CU + CU_OFFSET) / NASA_REFERENCE)
        else:
            ratio = COSMIC_LIFESPAN / CONVERGENCE_YEAR
            cu_diff = (total_seconds * ratio) / SECONDS_PER_YEAR
            if compression_model.lower() == 'polynomial' and total_seconds < Decimal(-31557600) * Decimal('1000'):
                cu_diff /= Decimal('1.0001')
            cu_time = BASE_CU + CU_OFFSET + cu_diff
            nasa_time = cu_time * Decimal('0.9997826')

        output = {
            'cu_time': f'{cu_time:.2f}',
            'nasa_time': f'{nasa_time:.2f}',
            'gregorian': parsed_time.strftime('%Y-%m-%d %H:%M:%S %Z')
        }
        logging.info(f"Converted {gregorian_time_str} to {output['cu_time']}")
        return jsonify(output)
    except Exception as e:
        logging.error(f"Error in gregorian_to_cu: {str(e)}")
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    # You can use port 5001 (or any port you prefer)
    app.run(host='0.0.0.0', port=5001)
