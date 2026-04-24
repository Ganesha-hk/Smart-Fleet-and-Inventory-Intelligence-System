import React from 'react';
import { motion } from 'framer-motion';

const MotionDiv = motion.div;

const InventoryCard = ({ card, index }) => {
  return (
    <MotionDiv
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.08 }}
      className="glass-card p-6"
    >
      <p className="text-gray-400 text-sm font-medium">{card.title}</p>
      <h3 className="text-2xl font-bold mt-1">{card.value}</h3>
      <p className="text-xs text-gray-500 mt-2 leading-5">{card.subtitle}</p>
    </MotionDiv>
  );
};

export default InventoryCard;
