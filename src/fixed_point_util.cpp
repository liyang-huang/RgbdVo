#include "fixed_point_util.hpp"
#include <opencv2/core/utility.hpp>
#include <iostream>
#include <gmpxx.h>
#include <gmp.h>
using namespace cv;

FixedPointScalar::FixedPointScalar(
	FIXP_SCALAR_TYPE value_floating,
	FixedPointConfig config) :
	FixedPointType<FIXP_SCALAR_TYPE, FIXP_INT_SCALAR_TYPE>(value_floating, 0, config) {
        if(cvIsNaN(value_floating))
           value = 0;
        else
	   value = static_cast<FIXP_INT_SCALAR_TYPE>(value_floating * (1LL << config.shift));
        mpz_set_si(big_value, value);
        //std::cout << "value_floating " << value_floating << std::endl;
        //std::cout << "value " << value << std::endl;
        //gmp_printf("%Zd\n", big_value);
	
        //check_bit_width(0);
        mpz_t max_value;
        mpz_init(max_value);
	mpz_set_si(max_value, (int64_t)1);
        mpz_mul_2exp(max_value, max_value, (config.bit_width - 1));
        mpz_t min_value;
        mpz_init(min_value);
	mpz_set_si(min_value, (int64_t)-1);
        mpz_mul_2exp(min_value, min_value, (config.bit_width - 1));
        int comp1 = mpz_cmp(max_value, big_value); // < max value
        int comp2 = mpz_cmp(min_value, big_value); // >= min value
        if(!((comp1 == 1) && (comp2 != 1)))
        {
             std::cout << "value_floating " << value_floating << std::endl;
             std::cout << "value " << value << std::endl;
             gmp_printf("%Zd\n", big_value);
             gmp_printf("%Zd\n", max_value);
             gmp_printf("%Zd\n", min_value);
             std::cout << "comp1" << comp1 << std::endl;
             std::cout << "comp2" << comp2 << std::endl;
        }
	assert(comp1 == 1);
	assert(comp2 != 1);
        mpz_clear(max_value);
        mpz_clear(min_value);
}
/*
FixedPointScalar::FixedPointScalar(
	FIXP_INT_SCALAR_TYPE value,
	FixedPointConfig config) :
	FixedPointType<FIXP_SCALAR_TYPE, FIXP_INT_SCALAR_TYPE>(0, value, config) {
	value_floating = FIXP_SCALAR_TYPE(value) / FIXP_SCALAR_TYPE(1LL << config.shift);
	check_bit_width(0);
}
*/
FixedPointScalar::FixedPointScalar(
	const FixedPointScalar &object): 
	FixedPointType<FIXP_SCALAR_TYPE, FIXP_INT_SCALAR_TYPE>(object){
	value_floating = object.value_floating; 
	value = object.value;
        mpz_set(big_value, object.big_value);
	config.sign = object.config.sign;
	config.bit_width = object.config.bit_width;
	config.shift = object.config.shift;
        //std::cout << "value_floating " << value_floating << std::endl;
        //std::cout << "value " << value << std::endl;
        //gmp_printf("%Zd\n", big_value);

	//check_bit_width(0);
        mpz_t max_value;
        mpz_init(max_value);
	mpz_set_si(max_value, (int64_t)1);
        mpz_mul_2exp(max_value, max_value, (config.bit_width - 1));
        mpz_t min_value;
        mpz_init(min_value);
	mpz_set_si(min_value, (int64_t)-1);
        mpz_mul_2exp(min_value, min_value, (config.bit_width - 1));
        int comp1 = mpz_cmp(max_value, big_value); // < max value
        int comp2 = mpz_cmp(min_value, big_value); // >= min value
        if(!((comp1 == 1) && (comp2 != 1)))
        {
             std::cout << "value_floating " << value_floating << std::endl;
             std::cout << "value_floating " << object.value_floating << std::endl;
             std::cout << "value " << value << std::endl;
             std::cout << "value " << object.value << std::endl;
             gmp_printf("%Zd\n", big_value);
             gmp_printf("%Zd\n", object.big_value);
             gmp_printf("%Zd\n", max_value);
             gmp_printf("%Zd\n", min_value);
             std::cout << "comp1" << comp1 << std::endl;
             std::cout << "comp2" << comp2 << std::endl;
        }
	assert(comp1 == 1);
	assert(comp2 != 1);
        mpz_clear(max_value);
        mpz_clear(min_value);
}

FixedPointVector::FixedPointVector(const FixedPointScalar &_x, const FixedPointScalar &_y, const FixedPointScalar &_z)
                                   : x(_x), y(_y), z(_z)
{}

void FixedPointScalar::check_bit_width(int op) {
	if (!enable_check_bit_width)
		return;

	if (config.sign == 0) {
		//debug
		//if (value >= (1LL << (config.bit_width)))
		//	printf("[FIXP_ERROR] shift (%i) value (%lli) >= 2^(bit_width) (%i)\n", config.shift, value, config.bit_width);
                //if(!(value < (1LL << (config.bit_width))))
                //{
                //    std::cout << "liyang value_floating" << value_floating << std::endl;
                //    std::cout << "liyang shift" << config.shift << std::endl;
                //    std::cout << "liyang value" << value << std::endl;
                //    std::cout << "liyang bitwidth" << config.bit_width << std::endl;
                //    std::cout << value << std::endl;
                //    std::cout << (1LL << config.bit_width) << std::endl;
                //}
	        //assert(value < (1LL << config.bit_width));
                mpz_t max_value;
                mpz_init(max_value);
		mpz_set_si(max_value, (int64_t)1);
                mpz_mul_2exp(max_value, max_value, config.bit_width);
                int comp0 = mpz_cmp(max_value, big_value); // < max value
		assert(comp0 == 1);
                mpz_clear(max_value);
	}
	else {
		//debug
		//if (value >= (1LL << (config.bit_width - 1)) || value < -(1LL << (config.bit_width - 1)))
		//	printf("[FIXP_ERROR] shift (%i) value (%lli) >= 2^(bit_width - 1) (%i)\n", config.shift, value, config.bit_width - 1);
                //if(!(value < (1LL << (config.bit_width - 1))))
                //if(!( value >= -(1LL << (config.bit_width - 1)) ))
                //{
                //    std::cout << "NaN " << cvIsNaN(value_floating) << std::endl;
                //    std::cout << "liyang value_floating " << value_floating << std::endl;
                //    std::cout << "liyang shift " << config.shift << std::endl;
                //    std::cout << "liyang value " << value << std::endl;
                //    std::cout << "liyang bitwidth " << config.bit_width << std::endl;
                //    std::cout << "liyang op " << op << std::endl;
                //}
	        //assert(value < (1LL << (config.bit_width - 1))); 
	        //assert(value >= -(1LL << (config.bit_width - 1)));
                mpz_t max_value;
                mpz_init(max_value);
		mpz_set_si(max_value, (int64_t)1);
                mpz_mul_2exp(max_value, max_value, (config.bit_width - 1));
                mpz_t min_value;
                mpz_init(min_value);
		mpz_set_si(min_value, (int64_t)-1);
                mpz_mul_2exp(min_value, min_value, (config.bit_width - 1));
                int comp1 = mpz_cmp(max_value, big_value); // < max value
                int comp2 = mpz_cmp(min_value, big_value); // >= min value
                if(!((comp1 == 1) && (comp2 != 1)))
                {
                     std::cout << "value_floating" << value_floating << std::endl;
                     gmp_printf("%Zd\n", big_value);
                     gmp_printf("%Zd\n", max_value);
                     gmp_printf("%Zd\n", min_value);
                     std::cout << "comp1" << comp1 << std::endl;
                     std::cout << "comp2" << comp2 << std::endl;
                     std::cout << "liyang op " << op << std::endl;
                }
		assert(comp1 == 1);
		assert(comp2 != 1);
                mpz_clear(max_value);
                mpz_clear(min_value);
	}
}

void FixedPointScalar::set_value(FIXP_INT_SCALAR_TYPE value, FixedPointConfig config){
	value_floating = FIXP_SCALAR_TYPE(value) / FIXP_SCALAR_TYPE(1LL << config.shift);
	this->value = value;
	this->config.sign = config.sign;
	this->config.bit_width = config.bit_width;
	this->config.shift = config.shift;
	check_bit_width(0);
}

void FixedPointScalar::set_bit_width(int bit_width){
	this->config.bit_width = bit_width;
	check_bit_width(0);
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
        mpz_set(big_value, object.big_value);
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
        mpz_add(return_object.big_value, big_value, object.big_value);
	return_object.config.sign = config.sign | object.config.sign;
	//return_object.config.bit_width = std::max(config.bit_width, object.config.bit_width) + 1;
	return_object.config.bit_width = config.bit_width, object.config.bit_width;
	return_object.check_bit_width(1);
	return return_object;
}

void FixedPointScalar::operator += (const FixedPointScalar &object) {
	assert(config.shift == object.config.shift);
	value_floating = value_floating + object.value_floating;
	value = value + object.value;
        mpz_add(big_value, big_value, object.big_value);
	config.sign = config.sign | object.config.sign;
	//config.bit_width = std::max(config.bit_width, object.config.bit_width) + 1;
	config.bit_width = config.bit_width;
	check_bit_width(1);
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
        mpz_sub(return_object.big_value, big_value, object.big_value);
	return_object.config.sign = 1;
	//return_object.config.bit_width = std::max(config.bit_width, object.config.bit_width) + 1;
	return_object.config.bit_width = config.bit_width;
	return_object.check_bit_width(2);
	return return_object;
}

void FixedPointScalar::operator -= (const FixedPointScalar &object) {
	assert(config.shift == object.config.shift);
	value_floating = value_floating - object.value_floating;
	value = value - object.value;
        mpz_sub(big_value, big_value, object.big_value);
	config.sign = 1;
	//config.bit_width = std::max(config.bit_width, object.config.bit_width) + 1;
	config.bit_width = config.bit_width;
	check_bit_width(2);
}

// Ref: https://www.geeksforgeeks.org/operator-overloading-c/
FixedPointScalar FixedPointScalar::operator * (const FixedPointScalar &object) {
	FixedPointScalar return_object(*this);
	return_object.value_floating = value_floating * object.value_floating;
	//return_object.value = value * object.value;
        int64_t temp1 = value;
        int64_t temp2 = object.value;
        int64_t temp3 = (temp1 * temp2) >> config.shift;
	return_object.value = temp3;
        mpz_mul(return_object.big_value, big_value, object.big_value);
        mpz_t shift_value;
        mpz_init(shift_value);
	mpz_set_si(shift_value, (int64_t)1);
        mpz_mul_2exp(shift_value, shift_value, config.shift);
        mpz_div(return_object.big_value, return_object.big_value, shift_value);
        mpz_clear(shift_value);
	return_object.config.sign = config.sign | object.config.sign;
	//return_object.config.shift = config.shift + object.config.shift;
	return_object.config.shift = config.shift;
	//return_object.config.bit_width = config.bit_width + object.config.bit_width;
	return_object.config.bit_width = config.bit_width;
        //std::cout << "value " << return_object.value << std::endl;
        //std::cout << "bit_width " << return_object.config.bit_width << std::endl;
        //std::cout << "value_floating " << value_floating << std::endl;
        //std::cout << "value_floating " << object.value_floating << std::endl;
        //std::cout << "value_floating " << return_object.value_floating << std::endl;
        //std::cout << "value " << value << std::endl;
        //std::cout << "value " << object.value << std::endl;
        //std::cout << "value " << return_object.value << std::endl;
        //gmp_printf("%Zd\n", big_value);
        //gmp_printf("%Zd\n", object.big_value);
        //gmp_printf("%Zd\n", return_object.big_value);

	//return_object.check_bit_width(3);
        mpz_t max_value;
        mpz_init(max_value);
        mpz_set_si(max_value, (int64_t)1);
        mpz_mul_2exp(max_value, max_value, (return_object.config.bit_width - 1));
        mpz_t min_value;
        mpz_init(min_value);
        mpz_set_si(min_value, (int64_t)-1);
        mpz_mul_2exp(min_value, min_value, (return_object.config.bit_width - 1));
        int comp1 = mpz_cmp(max_value, return_object.big_value); // < max value
        int comp2 = mpz_cmp(min_value, return_object.big_value); // >= min value
        if(!((comp1 == 1) && (comp2 != 1)))
        {
             std::cout << "value_floating " << value_floating << std::endl;
             std::cout << "value_floating " << object.value_floating << std::endl;
             std::cout << "value_floating " << return_object.value_floating << std::endl;
             std::cout << "value " << value << std::endl;
             std::cout << "value " << object.value << std::endl;
             std::cout << "value " << return_object.value << std::endl;
             gmp_printf("%Zd\n", big_value);
             gmp_printf("%Zd\n", object.big_value);
             gmp_printf("%Zd\n", return_object.big_value);
             gmp_printf("%Zd\n", max_value);
             gmp_printf("%Zd\n", min_value);
             std::cout << "comp1" << comp1 << std::endl;
             std::cout << "comp2" << comp2 << std::endl;
        }
        assert(comp1 == 1);
        assert(comp2 != 1);
        mpz_clear(max_value);
        mpz_clear(min_value);
	return return_object;
}

// Ref: https://www.geeksforgeeks.org/operator-overloading-c/
FixedPointScalar FixedPointScalar::operator / (const FixedPointScalar &object) {
	FixedPointScalar return_object(*this);
	return_object.value_floating = value_floating / object.value_floating;
	//return_object.value = value / object.value;
        int64_t temp1 = value;
        int64_t temp2 = object.value;
        int64_t temp3 = (temp1 << config.shift) / temp2;
	return_object.value = temp3;
        mpz_t shift_value;
        mpz_init(shift_value);
	mpz_set_si(shift_value, (int64_t)1);
        mpz_mul_2exp(shift_value, shift_value, config.shift);
        mpz_mul(shift_value, big_value, shift_value);
        mpz_div(return_object.big_value, shift_value, object.big_value); 
        mpz_clear(shift_value);
	return_object.config.sign = config.sign | object.config.sign;
	//return_object.config.shift = config.shift - object.config.shift;
	return_object.config.bit_width = config.bit_width;
	return_object.check_bit_width(4);
	return return_object;
}
/*
FixedPointScalar FixedPointScalar::operator >> (int bit_shift) {
	FixedPointScalar return_object(*this);
	return_object.value_floating = value_floating;
	return_object.value = value >> bit_shift;
	return_object.config.shift = config.shift - bit_shift;
	return_object.config.bit_width = config.bit_width - bit_shift;
	return_object.check_bit_width(5);
	return return_object;
}

FixedPointScalar FixedPointScalar::operator << (int bit_shift) {
	FixedPointScalar return_object(*this);
	return_object.value_floating = value_floating;
	return_object.value = value << bit_shift;
	return_object.config.shift = config.shift + bit_shift;
	return_object.config.bit_width = config.bit_width + bit_shift;
	return_object.check_bit_width(5);
	return return_object;
}
*/
FixedPointScalar FixedPointScalar::sqrt() {
	FixedPointScalar return_object(*this);
	return_object.value_floating = std::sqrt(value_floating);
        int64_t temp1 = value << config.shift;
	return_object.value = FIXP_INT_SCALAR_TYPE(std::floor(std::sqrt(temp1)));
        mpz_t shift_value;
        mpz_init(shift_value);
	mpz_set_si(shift_value, (int64_t)1);
        mpz_mul_2exp(shift_value, shift_value, config.shift);
        mpz_mul(shift_value, big_value, shift_value);
        mpz_sqrt(return_object.big_value, shift_value);
        mpz_clear(shift_value);
	//return_object.value = FIXP_INT_SCALAR_TYPE(std::floor(std::sqrt(value))) << int(config.shift / 2);
	//return_object.config.bit_width = int((config.bit_width + 1) / 2);
	return_object.config.bit_width = config.bit_width;
	//assert(return_object.config.shift % 2 == 0);
	//return_object.config.shift = int(config.shift / 2);
	return_object.config.shift = config.shift;
	return_object.check_bit_width(6);
	return return_object;
}

FixedPointScalar FixedPointScalar::abs() {
	FixedPointScalar return_object(*this);
	return_object.value_floating = std::abs(value_floating);
	return_object.value = std::abs(value);
        mpz_abs(return_object.big_value, big_value);
	//return_object.config.sign = 0;
	return_object.config.sign = config.sign;
	return_object.check_bit_width(7);
	return return_object;
}

FIXP_SCALAR_TYPE FixedPointScalar::to_floating() {
        //std::cout << "value " << value << std::endl;
	//return FIXP_SCALAR_TYPE(value) / FIXP_SCALAR_TYPE(1LL << config.shift);

        int temp =  mpz_fits_slong_p(big_value);

        if((value_floating!=(FIXP_SCALAR_TYPE)0.0) && (temp == (int)0))
        {
                std::cout << "to_floating() overflow " << std::endl;
                exit(1);
        }

        int64_t big_value_chg = mpz_get_si(big_value);
        double value_double = (double)big_value_chg / (double)(1LL << config.shift);
        return (FIXP_SCALAR_TYPE)value_double;
}

void FixedPointScalar::print_big_value() {
      gmp_printf("%Zd\n", big_value);
}

/*
FixedPointMatrix::FixedPointMatrix(
	std::vector<FixedPointScalar> scalar_vector,
	int rows,
	int cols) :
	FixedPointType<FIXP_MATRIX_TYPE, FIXP_INT_MATRIX_TYPE>(
		FIXP_MATRIX_TYPE::Zero(rows, cols), 
		FIXP_INT_MATRIX_TYPE::Zero(rows, cols), 
		scalar_vector[0].config) {
	assert(int(scalar_vector.size()) == rows * cols);
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			FixedPointScalar temp = scalar_vector[r*cols + c];
			value_floating(r, c) = temp.value_floating;
			value(r, c) = temp.value;
			//assert(config == temp.config);
		}
	}
	check_bit_width();
}

FixedPointMatrix::FixedPointMatrix(
	const FixedPointMatrix &object) :
	FixedPointType<FIXP_MATRIX_TYPE, FIXP_INT_MATRIX_TYPE>(object) {
	value_floating = object.value_floating;
	value = object.value;
	config.sign = object.config.sign;
	config.bit_width = object.config.bit_width;
	config.shift = object.config.shift;
	check_bit_width();
}

FixedPointMatrix::FixedPointMatrix(
	FixedPointMatrix&& object) :
        FixedPointType<FIXP_MATRIX_TYPE, FIXP_INT_MATRIX_TYPE>(object) {
    value_floating = std::move(object.value_floating);
    value = std::move(object.value);
    config = object.config;
	check_bit_width();
}

FixedPointMatrix& FixedPointMatrix::operator= (FixedPointMatrix& object){
    value_floating = object.value_floating;
    value = object.value;
    config = object.config;
    return *this;
};

FixedPointMatrix& FixedPointMatrix::operator= (FixedPointMatrix&& object){
    value_floating = std::move(object.value_floating);
    value = std::move(object.value);
    config = object.config;
    return *this;
};


void FixedPointMatrix::check_bit_width() {
	if (!enable_check_bit_width)
		return;

	FIXP_INT_SCALAR_TYPE max_value = value.maxCoeff();
	FIXP_INT_SCALAR_TYPE min_value = value.minCoeff();

	if (config.sign == 0) {
		FIXP_INT_SCALAR_TYPE capacity = 1LL << (config.bit_width);
		//debug
		if (max_value >= capacity || min_value < 0)
			//printf("[FIXP_ERROR] shift (%i) max_value (%lli) >= 2^(bit_width) (%i) or min_value (%lli) < 0\n", config.shift, max_value, config.bit_width, min_value);
			printf("[FIXP_ERROR] shift (%d) max_value (%ld) >= 2^(bit_width) (%d) or min_value (%ld) < 0\n", config.shift, max_value, config.bit_width, min_value);
		assert(max_value < capacity);
		assert(min_value >= 0);
	}
	else {
		FIXP_INT_SCALAR_TYPE capacity = 1LL << (config.bit_width - 1);
		//debug
		if (std::abs(max_value) >= capacity || std::abs(min_value) >= capacity)
			//printf("[FIXP_ERROR] shift (%i) max_value (%lli) or abs(min_value) (%lli) >= 2^(bit_width - 1) (%i)\n", config.shift, max_value, min_value, config.bit_width - 1);
			printf("[FIXP_ERROR] shift (%d) max_value (%ld) or abs(min_value) (%ld) >= 2^(bit_width - 1) (%d)\n", config.shift, max_value, min_value, config.bit_width - 1);
		assert(std::abs(max_value) < capacity);
		assert(std::abs(min_value) < capacity);
	}
}

void FixedPointMatrix::assign(
	const FixedPointScalar &object,
	const int& row,
	const int& col) {
	assert(config.sign == object.config.sign);
	assert(config.shift == object.config.shift);
	assert(config.bit_width >= object.config.bit_width);
	value_floating(row, col) = object.value_floating;
	value(row, col) = object.value;
}

FixedPointScalar FixedPointMatrix::operator()(
	const int& row,
	const int& col) {
	FixedPointScalar scalar(value_floating(row, col), config);
	scalar.set_value(value(row, col), config);
	return scalar;
}

FIXP_MATRIX_TYPE FixedPointMatrix::to_floating() {
	FIXP_MATRIX_TYPE return_floating = value.cast<FIXP_SCALAR_TYPE>();
	return return_floating / (1LL << config.shift);
}

std::vector<FixedPointScalar> FixedPointMatrix::to_vector() const {
	std::vector<FixedPointScalar> ret(value.rows()*value.cols(), FixedPointScalar((FIXP_SCALAR_TYPE)0));
	for (int r = 0; r < value.rows(); r++) {
		for (int c = 0; c < value.cols(); c++) {
			FixedPointScalar temp(value_floating(r, c), config);
			temp.value = value(r, c);
			temp.check_bit_width();
			ret[r*value.cols() + c] = temp;
		}
	}
	return ret;
}


*/






std::vector<FixedPointScalar> f_Mat2Vec(const Mat& in_mat, FixedPointConfig config) {
    if(in_mat.depth() != CV_32F)
        CV_Error(Error::StsBadSize, "Input Mat depth has to be FIXP_SCALAR_TYPE in Mat2Vec.");
    int rows = in_mat.rows;
    int cols = in_mat.cols;

    std::vector<FixedPointScalar> out_vec;

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            FIXP_SCALAR_TYPE value = in_mat.at<FIXP_SCALAR_TYPE>(r, c);
            FixedPointScalar temp(value, config);
            out_vec.push_back(temp);
        }
    }

    return out_vec;
}

std::vector<FixedPointScalar> i_Mat2Vec(const Mat& in_mat, FixedPointConfig config) {
    int rows = in_mat.rows;
    int cols = in_mat.cols;

    std::vector<FixedPointScalar> out_vec;

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            FIXP_INT_SCALAR_TYPE value = in_mat.at<FIXP_INT_SCALAR_TYPE>(r, c);
            FixedPointScalar temp(value, config);
            out_vec.push_back(temp);
        }
    }

    return out_vec;
}

Mat Vec2Mat_d(const std::vector<FixedPointScalar>& in_vec, int rows, int cols) {

    Mat out_mat;
    out_mat.create(rows, cols, CV_64FC1);
  
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            FixedPointScalar temp = in_vec[r*cols + c];
            out_mat.at<double>(r, c) = (double)temp.to_floating();
        }
    }

    return out_mat;
}

Mat Vec2Mat_f(const std::vector<FixedPointScalar>& in_vec, int rows, int cols) {

    Mat out_mat;
    out_mat.create(rows, cols, CV_32FC1);
  
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            FixedPointScalar temp = in_vec[r*cols + c];
            out_mat.at<FIXP_SCALAR_TYPE>(r, c) = temp.to_floating();
            //out_mat.at<FIXP_SCALAR_TYPE>(r, c) = temp.value_floating;
        }
    }

    return out_mat;
}

Mat Vec2Mat_i(const std::vector<FixedPointScalar>& in_vec, int rows, int cols) {

    Mat out_mat;
    out_mat.create(rows, cols, CV_32SC1);
  
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            FixedPointScalar temp = in_vec[r*cols + c];
            out_mat.at<FIXP_INT_SCALAR_TYPE>(r, c) = temp.value;
        }
    }

    return out_mat;
}

Mat PVec2Mat_i(const std::vector<FixedPointVector>& in_vec, int rows, int cols) {

    Mat out_mat;
    out_mat.create(rows, cols, CV_32SC3);
  
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            FixedPointVector temp = in_vec[r*cols + c];
            out_mat.at<Vec3i>(r, c)[0] = temp.x.value;
            out_mat.at<Vec3i>(r, c)[1] = temp.y.value;
            out_mat.at<Vec3i>(r, c)[2] = temp.z.value;
        }
    }

    return out_mat;
}

Mat PVec2Mat_f(const std::vector<FixedPointVector>& in_vec, int rows, int cols) {

    Mat out_mat;
    out_mat.create(rows, cols, CV_32FC3);
  
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            FixedPointVector temp = in_vec[r*cols + c];
            out_mat.at<Point3f>(r, c).x = temp.x.value_floating;
            out_mat.at<Point3f>(r, c).y = temp.y.value_floating;
            out_mat.at<Point3f>(r, c).z = temp.z.value_floating;
        }
    }

    return out_mat;
}

std::vector<FixedPointVector> i_PMat2Vec(const Mat& in_mat, FixedPointConfig config) {
    int rows = in_mat.rows;
    int cols = in_mat.cols;

    std::vector<FixedPointVector> out_vec;

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            FIXP_INT_SCALAR_TYPE value_x = in_mat.at<Vec3i>(r, c)[0];
            FIXP_INT_SCALAR_TYPE value_y = in_mat.at<Vec3i>(r, c)[1];
            FIXP_INT_SCALAR_TYPE value_z = in_mat.at<Vec3i>(r, c)[2];
            FixedPointScalar temp_x(value_x, config);
            FixedPointScalar temp_y(value_y, config);
            FixedPointScalar temp_z(value_z, config);
            FixedPointVector temp(temp_x, temp_y, temp_z);
            out_vec.push_back(temp);
        }
    }

    return out_vec;
}

std::vector<FixedPointVector> f_PMat2Vec(const Mat& in_mat, FixedPointConfig config) {
    int rows = in_mat.rows;
    int cols = in_mat.cols;

    std::vector<FixedPointVector> out_vec;

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            FIXP_SCALAR_TYPE value_x = in_mat.at<Vec3f>(r, c)[0];
            FIXP_SCALAR_TYPE value_y = in_mat.at<Vec3f>(r, c)[1];
            FIXP_SCALAR_TYPE value_z = in_mat.at<Vec3f>(r, c)[2];
            FixedPointScalar temp_x(value_x, config);
            FixedPointScalar temp_y(value_y, config);
            FixedPointScalar temp_z(value_z, config);
            FixedPointVector temp(temp_x, temp_y, temp_z);
            out_vec.push_back(temp);
        }
    }

    return out_vec;
}

