#ifndef ENTITY_GEOMETRY_HPP_
#define ENTITY_GEOMETRY_HPP_

#include <silkroad_lib/position.h>

#include <memory>
#include <optional>

namespace entity {

class Geometry {
public:
  virtual std::optional<double> timeUntilEnter(const sro::Position &srcPos, const sro::Position &destPos, float movementSpeed) const = 0;
  virtual std::optional<double> timeUntilEnter(const sro::Position &srcPos, sro::Angle movementAngle, float movementSpeed) const = 0;
  virtual std::optional<double> timeUntilExit(const sro::Position &srcPos, const sro::Position &destPos, float movementSpeed) const = 0;
  virtual std::optional<double> timeUntilExit(const sro::Position &srcPos, sro::Angle movementAngle, float movementSpeed) const = 0;
  virtual std::unique_ptr<Geometry> clone() const = 0;
  virtual bool pointIsInside(const sro::Position &point) const = 0;
};

class Circle : public Geometry {
public:
  Circle(const sro::Position &center, double radius);
  virtual std::optional<double> timeUntilEnter(const sro::Position &srcPos, const sro::Position &destPos, float movementSpeed) const override;
  virtual std::optional<double> timeUntilEnter(const sro::Position &srcPos, sro::Angle movementAngle, float movementSpeed) const override;
  virtual std::optional<double> timeUntilExit(const sro::Position &srcPos, const sro::Position &destPos, float movementSpeed) const override;
  virtual std::optional<double> timeUntilExit(const sro::Position &srcPos, sro::Angle movementAngle, float movementSpeed) const override;
  virtual std::unique_ptr<Geometry> clone() const override;
  virtual bool pointIsInside(const sro::Position &point) const override;

  const sro::Position& center() const;
  double radius() const;
private:
  const sro::Position center_;
  const double radius_;
};

class Rectangle : public Geometry {
public:
  Rectangle(const sro::Position &topLeft, const sro::Position &topRight, const sro::Position &bottomRight, const sro::Position &bottomLeft);
  virtual std::optional<double> timeUntilEnter(const sro::Position &srcPos, const sro::Position &destPos, float movementSpeed) const override;
  virtual std::optional<double> timeUntilEnter(const sro::Position &srcPos, sro::Angle movementAngle, float movementSpeed) const override;
  virtual std::optional<double> timeUntilExit(const sro::Position &srcPos, const sro::Position &destPos, float movementSpeed) const override;
  virtual std::optional<double> timeUntilExit(const sro::Position &srcPos, sro::Angle movementAngle, float movementSpeed) const override;
  virtual std::unique_ptr<Geometry> clone() const override;
  virtual bool pointIsInside(const sro::Position &point) const override;
private:
  const sro::Position topLeft_;
  const sro::Position topRight_;
  const sro::Position bottomRight_;
  const sro::Position bottomLeft_;
};

} // namespace entity

#endif // ENTITY_GEOMETRY_HPP_