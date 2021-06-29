#include <opencv2/core.hpp>
#include <vector>

typedef int64_t FIXP_INT_SCALAR_TYPE;
typedef float FIXP_SCALAR_TYPE;

/**
 * Fixed point data configuration.
 * Include sign, bit_width, shift.
 */
class FixedPointConfig {
public:

	FixedPointConfig(
		int sign = 0,
		int bit_width = 0,
		int shift = 0) :
		sign(sign),
		bit_width(bit_width),
		shift(shift) {};

	// Copy constructor
	FixedPointConfig(const FixedPointConfig &object) {
		sign = object.sign;
		bit_width = object.bit_width;
		shift = object.shift;
	};

    // assignment
    FixedPointConfig& operator= (const FixedPointConfig& object) {
		sign = object.sign;
		bit_width = object.bit_width;
		shift = object.shift;
        return *this;
    };

	// destructor
	~FixedPointConfig() {};

	bool operator==(const FixedPointConfig& object) {
		if (sign == object.sign && bit_width == object.bit_width && shift == object.shift)
			return true;
		else
			return false;
	}

	int sign; // variable can be negative or not
	int bit_width; // max value < 2^bit_width
	int shift;  // 2^shift larger than float version
};

/**
 * Fixed point data type template.
 */
template<typename Floating, typename FixedPoint>
class FixedPointType {
public:

	FixedPointType(
		Floating value_floating,
		FixedPoint value,
		FixedPointConfig config = FixedPointConfig(0, 0, 0)) :
		value_floating(value_floating),
		value(value),
		config(config) {};

	// Copy constructor
	FixedPointType(const FixedPointType &object) {
		value_floating = object.value_floating;
		value = object.value;
		config = object.config;
	};

	bool enable_check_bit_width = true;
	Floating value_floating;
	FixedPoint value;
	FixedPointConfig config;
};

/**
 * Fixed point Scalar data type.
 */
class FixedPointScalar: public FixedPointType<FIXP_SCALAR_TYPE, FIXP_INT_SCALAR_TYPE> {
public:

	FixedPointScalar(
		FIXP_SCALAR_TYPE value_floating = 0,
		FixedPointConfig config = FixedPointConfig(0, 0, 0));

	// Copy constructor
	FixedPointScalar(const FixedPointScalar &object);

	// destructor
	~FixedPointScalar() {};

	void check_bit_width();

	void set_value(FIXP_INT_SCALAR_TYPE value, FixedPointConfig config);

	void set_bit_width(int bit_width);

	// Overload operator =
	FixedPointScalar& operator= (const FixedPointScalar &object);

	// Overload operator +
	FixedPointScalar operator + (const FixedPointScalar &object);
	
	// Overload operator +=
	void operator += (const FixedPointScalar &object);

	// Overload operator -
	FixedPointScalar operator - (const FixedPointScalar &object);

	// Overload operator -=
	void operator -= (const FixedPointScalar &object);

	// Overload operator *
	FixedPointScalar operator * (const FixedPointScalar &object);

	// Overload operator /
	FixedPointScalar operator / (const FixedPointScalar &object);

	// Overload operator >>
	FixedPointScalar operator >> (int bit_shift);

	// Overload operator <<
	FixedPointScalar operator << (int bit_shift);

	FixedPointScalar sqrt();

	FixedPointScalar abs();

	FIXP_SCALAR_TYPE to_floating();
};



