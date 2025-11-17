//package com.qczy.common.exception;
//
//import org.slf4j.Logger;
//import org.slf4j.LoggerFactory;
//import org.springframework.web.bind.annotation.ControllerAdvice;
//import org.springframework.web.bind.annotation.ExceptionHandler;
//import org.springframework.web.bind.annotation.ResponseBody;
//
///**
// * @description: 自定义异常处理
// * @author: DT
// * @date: 2021/4/19 21:17
// * @version: v1.0
// */
//
//@ControllerAdvice
//public class MyExceptionHandler {
//
//    private static final Logger logger = LoggerFactory.getLogger(MyExceptionHandler.class);
//
//    @ExceptionHandler(value =Exception.class)
//    @ResponseBody
//    public String exceptionHandler(Exception e){
//        logger.error("全局异常捕获>>>:" + e);
//        return "全局异常捕获,错误原因>>>"+e.getMessage();
//    }
//}
//
