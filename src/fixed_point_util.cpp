#include "fixed_point_util.h"

FixedPointScalar::FixedPointScalar(
	FIXP_SCALAR_TYPE value_floating,
	FixedPointConfig config) :
	FixedPointType<FIXP_SCALAR_TYPE, FIXP_INT_SCALAR_TYPE>(value_floating, 0, config) {
	value = static_cast<FIXP_INT_SCALAR_TYPE>(value_floating * (1LL << config.shift));
	check_bit_width();
}

FixedPointScalar::FixedPointScalar(
	const FixedPointScalar &object): 
	FixedPointType<FIXP_SCALAR_TYPE, FIXP_INT_SCALAR_TYPE>(object){
	value_floating = object.value_floating; 
	value = object.value;
	config.sign = object.config.sign;
	config.bit_width = object.config.bit_width;
	config.shift = object.config.shift;
	check_bit_width();
}

void FixedPointScalar::check_bit_width() {
	if (!enable_check_bit_width)
		return;

	if (config.sign == 0) {
		//debug
		//if (value >= (1LL << (config.bit_width)))
		//	printf("[FIXP_ERROR] shift (%i) value (%lli) >= 2^(bit_width) (%i)\n", config.shift, value, config.bit_width);
		assert(value < (1LL << config.bit_width));
	}
	else {
		//debug
		//if (value >= (1LL << (config.bit_width - 1)) || value < -(1LL << (config.bit_width - 1)))
		//	printf("[FIXP_ERROR] shift (%i) value (%lli) >= 2^(bit_width - 1) (%i)\n", config.shift, value, config.bit_width - 1);
		assert(value < (1LL << (config.bit_width - 1))); 
		assert(value >= -(1LL << (config.bit_width - 1)));
	}
}

void FixedPointScalar::set_value(FIXP_INT_SCALAR_TYPE value, FixedPointConfig config){
	value_floating = FIXP_SCALAR_TYPE(value) / FIXP_SCALAR_TYPE(1LL << config.shift);
	this->value = value;
	this->config.sign = config.sign;
	this->config.bit_width = config.bit_width;
	this->config.shift = config.shift;
	check_bit_width();
}

void FixedPointScalar::set_bit_width(int bit_width){
	this->config.bit_width = bit_width;
	check_bit_width();
}

// Ref: https://www.learncpp.com/cpp-tutorial/overloading-the-assignment-operator/
FixedPointScalar& FixedPointScalar::operator= (const FixedPointScalar &object)
{
	// self-assignment guard
	if (this == &object)
		return *this;

	// do the copy
	value_floating = object.value_floating;
	value = object.value;
	config.sign = object.config.sign;
	config.bit_width = object.config.bit_width;
	config.shift = object.config.shift;

	// return the existing object so we can chain this operator
	return *this;
}

// Ref: https://www.geeksforgeeks.org/operator-overloading-c/
FixedPointScalar FixedPointScalar::operator + (const FixedPointScalar &object) {
	assert(config.shift == object.config.shift);
	FixedPointScalar return_object(*this);
	return_object.value_floating = value_floating + object.value_floating;
	return_object.value = value + object.value;
	return_object.config.sign = config.sign | object.config.sign;
	return_object.config.bit_width = std::max(config.bit_width, object.config.bit_width) + 1;
	return_object.check_bit_width();
	return return_object;
}

void FixedPointScalar::operator += (const FixedPointScalar &object) {
	assert(config.shift == object.config.shift);
	value_floating = value_floating + object.value_floating;
	value = value + object.value;
	config.sign = config.sign | object.config.sign;
	config.bit_width = std::max(config.bit_width, object.config.bit_width) + 1;
	check_bit_width();
}

// Ref: https://www.geeksforgeeks.org/operator-overloading-c/
FixedPointScalar FixedPointScalar::operator - (const FixedPointScalar &object) {
	//debug
	//if (shift != object.shift)
	//	printf("[FIXP_ERROR] shift (%i) != object.shift (%i)\n", shift, object.shift);
	assert(config.shift == object.config.shift);
	FixedPointScalar return_object(*this);
	return_object.value_floating = value_floating - object.value_floating;
	return_object.value = value - object.value;
	return_object.config.sign = 1;
	return_object.config.bit_width = std::max(config.bit_width, object.config.bit_width) + 1;
	return_object.check_bit_width();
	return return_object;
}

void FixedPointScalar::operator -= (const FixedPointScalar &object) {
	assert(config.shift == object.config.shift);
	value_floating = value_floating - object.value_floating;
	value = value - object.value;
	config.sign = 1;
	config.bit_width = std::max(config.bit_width, object.config.bit_width) + 1;
	check_bit_width();
}

// Ref: https://www.geeksforgeeks.org/operator-overloading-c/
FixedPointScalar FixedPointScalar::operator * (const FixedPointScalar &object) {
	FixedPointScalar return_object(*this);
	return_object.value_floating = value_floating * object.value_floating;
	return_object.value = value * object.value;
	return_object.config.sign = config.sign | object.config.sign;
	return_object.config.shift = config.shift + object.config.shift;
	return_object.config.bit_width = config.bit_width + object.config.bit_width;
	return_object.check_bit_width();
	return return_object;
}

// Ref: https://www.geeksforgeeks.org/operator-overloading-c/
FixedPointScalar FixedPointScalar::operator / (const FixedPointScalar &object) {
	FixedPointScalar return_object(*this);
	return_object.value_floating = value_floating / object.value_floating;
	return_object.value = value / object.value;
	return_object.config.sign = config.sign | object.config.sign;
	return_object.config.shift = config.shift - object.config.shift;
	return_object.check_bit_width();
	return return_object;
}

FixedPointScalar FixedPointScalar::operator >> (int bit_shift) {
	FixedPointScalar return_object(*this);
	return_object.value_floating = value_floating;
	return_object.value = value >> bit_shift;
	return_object.config.shift = config.shift - bit_shift;
	return_object.config.bit_width = config.bit_width - bit_shift;
	return_object.check_bit_width();
	return return_object;
}

FixedPointScalar FixedPointScalar::operator << (int bit_shift) {
	FixedPointScalar return_object(*this);
	return_object.value_floating = value_floating;
	return_object.value = value << bit_shift;
	return_object.config.shift = config.shift + bit_shift;
	return_object.config.bit_width = config.bit_width + bit_shift;
	return_object.check_bit_width();
	return return_object;
}

FixedPointScalar FixedPointScalar::sqrt() {
	FixedPointScalar return_object(*this);
	return_object.value_floating = std::sqrt(value_floating);
	return_object.value = FIXP_INT_SCALAR_TYPE(std::floor(std::sqrt(value)));
	return_object.config.bit_width = int((config.bit_width + 1) / 2);
	assert(return_object.config.shift % 2 == 0);
	return_object.config.shift = int(config.shift / 2);
	return_object.check_bit_width();
	return return_object;
}

FixedPointScalar FixedPointScalar::abs() {
	FixedPointScalar return_object(*this);
	return_object.value_floating = std::abs(value_floating);
	return_object.value = std::abs(value);
	return_object.config.sign = 0;
	return_object.check_bit_width();
	return return_object;
}

FIXP_SCALAR_TYPE FixedPointScalar::to_floating() {
	return FIXP_SCALAR_TYPE(value) / FIXP_SCALAR_TYPE(1LL << config.shift);
}


