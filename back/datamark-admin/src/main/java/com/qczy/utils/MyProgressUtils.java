package com.qczy.utils;

import cn.hutool.core.util.NumberUtil;
import com.qczy.mapper.DataSonMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.math.BigDecimal;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/12/25 9:57
 * @Description:
 */
public class MyProgressUtils {


    public static int calculateCount(Integer currentCount, Integer sumCount) {
        return NumberUtil.div(currentCount.toString(), sumCount.toString(), 2).multiply(BigDecimal.valueOf(100)).intValue();
    }


    public static void main(String[] args) {
        System.out.println(calculateCount(2, 5));
    }

}
